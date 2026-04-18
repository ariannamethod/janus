/*
 * infer_janus_sonar_chain.c — Janus bidirectional chain inference.
 *
 * 8-step chain with calendar-drift compass (forward vs backward ratio),
 * Schumann resonance temperature modulation, best-of-3 per step,
 * destiny EMA across the chain, and SPA (Sentence Phonon Attention)
 * reseed of the weakest sentence at the end.
 *
 * For microjanus with dual weights — no MetaWeights, no chambers.
 * Coherence scored by unique-token ratio + length bonus.
 *
 *   make infer_janus_sonar_chain
 *   ./infer_janus_sonar_chain janus_sonar.bin "seed text"
 */
#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define DIM       128
#define NLAYERS   4
#define NHEADS    4
#define HEAD_DIM  32
#define HIDDEN    256
#define CTX       128
#define VOCAB     2048

#define CHAIN_STEPS    8       /* 3 backward + 5 forward typical */
#define SENT_MAX       200     /* max tokens per generated sentence */
#define SENT_MIN_LEN   24      /* no early cutoff before this length — forces full thoughts */
#define CAND_N         3       /* best-of-N candidates per step */
#define REP_WINDOW     64      /* repetition penalty window */
#define REP_PENALTY    0.65f   /* multiply seen-token logits by this */

/* ── AML physics state (field) ── */
/* Port of core/ariannamethod.c logit transformations: destiny, suffering,
   laws (entropy floor + resonance ceiling), prophecy debt accumulation.
   All six Kuramoto chambers modulate force coefficients in real time. */
typedef struct {
    float destiny_bias;        /* [0,1] — max-suppression strength */
    float pain;                /* [0,1] — compresses toward mean */
    float entropy_floor;       /* [0,1] — cap on max-vs-second gap */
    float resonance_ceiling;   /* [0,1] — additional gap cap */
    float prophecy_debt;       /* accumulated: (max-chosen)/(+1) per step */
    float debt_decay;          /* per-step multiplier on debt */
    /* Chambers: Kuramoto 6-oscillator ring */
    float ch_act[6];
    float ch_decay[6];
    float ch_coup[6][6];
} AMLField;

enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_CMPLX };

static void aml_init(AMLField* f) {
    memset(f, 0, sizeof(*f));
    f->destiny_bias     = 0.20f;
    f->pain             = 0.0f;
    f->entropy_floor    = 0.10f;
    f->resonance_ceiling= 0.95f;
    f->prophecy_debt    = 0.0f;
    f->debt_decay       = 0.998f;
    /* Chamber initial activations (LOVE + FLOW slight bias — ready posture) */
    f->ch_act[CH_LOVE] = 0.2f; f->ch_act[CH_FLOW] = 0.15f;
    static const float decay[6] = {0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f};
    memcpy(f->ch_decay, decay, sizeof(decay));
    /* Coupling matrix (antisymmetric-ish) from core/ariannamethod.c */
    static const float coup[6][6] = {
        { 0.00f,-0.30f, 0.50f, 0.40f,-0.20f, 0.10f},
        {-0.30f, 0.00f,-0.40f,-0.50f, 0.50f, 0.20f},
        { 0.50f,-0.30f, 0.00f, 0.20f,-0.30f, 0.30f},
        { 0.40f,-0.50f, 0.30f, 0.00f,-0.30f, 0.40f},
        {-0.20f, 0.40f,-0.20f,-0.30f, 0.00f, 0.30f},
        { 0.10f, 0.20f, 0.30f, 0.40f, 0.30f, 0.00f}
    };
    memcpy(f->ch_coup, coup, sizeof(coup));
}

/* Kuramoto crossfire step: act[i] += K·sum_j(coup[i][j]·sin(act[j]-act[i])), then decay */
static void aml_chambers_xfire(AMLField* f, int iters) {
    for (int t = 0; t < iters; t++) {
        float old[6]; memcpy(old, f->ch_act, sizeof(old));
        for (int i = 0; i < 6; i++) {
            f->ch_act[i] *= f->ch_decay[i];
            for (int j = 0; j < 6; j++)
                if (i != j) f->ch_act[i] += 0.03f * f->ch_coup[i][j] * sinf(old[j] - old[i]);
            if (f->ch_act[i] < 0) f->ch_act[i] = 0;
            if (f->ch_act[i] > 1) f->ch_act[i] = 1;
        }
    }
}

/* Destiny: suppress below-max logits by (max - logits[i]) * bias * 0.5
   Result: distribution becomes more peaked around the current dominant token. */
static void aml_apply_destiny(float* logits, int n, float bias) {
    if (n <= 0 || bias < 0.001f) return;
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) logits[i] -= (mx - logits[i]) * bias * 0.5f;
}

/* Suffering: pain compresses logits toward mean (blunts both peaks and valleys) */
static void aml_apply_suffering(float* logits, int n, float pain) {
    if (n <= 0 || pain < 0.01f) return;
    float mean = 0; for (int i = 0; i < n; i++) mean += logits[i]; mean /= (float)n;
    float factor = 1.0f - 0.5f * pain;
    for (int i = 0; i < n; i++) logits[i] = mean + (logits[i] - mean) * factor;
}

/* Laws: entropy floor + resonance ceiling — cap on max-vs-second gap */
static void aml_apply_laws(float* logits, int n, float ent_floor, float res_ceil) {
    if (n <= 0) return;
    float mx = logits[0], sec = -1e30f;
    for (int i = 1; i < n; i++) {
        if (logits[i] > mx) { sec = mx; mx = logits[i]; }
        else if (logits[i] > sec) sec = logits[i];
    }
    float gap = mx - sec;
    if (gap > 0 && ent_floor > 0) {
        float max_gap = (1.0f - ent_floor) * 10.0f;
        if (gap > max_gap) {
            float reduce = (gap - max_gap) * 0.5f;
            for (int i = 0; i < n; i++) if (logits[i] == mx) logits[i] -= reduce;
        }
    }
    if (res_ceil < 1.0f) {
        float ceiling_gap = res_ceil * 10.0f;
        float new_gap = mx - sec;
        if (new_gap > ceiling_gap) {
            float reduce = (new_gap - ceiling_gap) * 0.3f;
            for (int i = 0; i < n; i++) if (logits[i] >= mx - 0.001f) logits[i] -= reduce;
        }
    }
}

/* Prophecy debt contribution of a choice: (max - chosen) / (diff + 1) */
static float aml_prophecy_debt(const float* logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0;
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float diff = mx - logits[chosen];
    return diff > 0 ? diff / (diff + 1.0f) : 0;
}

/* Apply full field pipeline + update prophecy debt after choice.
   Chamber modulation: destiny bias amplified by VOID, suppressed by FEAR.
                       pain amplified by RAGE.
                       Laws always on. */
static void aml_apply_field(float* logits, int n, const AMLField* f) {
    float dest = f->destiny_bias * (1.0f + 0.6f*f->ch_act[CH_VOID] - 0.3f*f->ch_act[CH_FEAR]);
    float pain = f->pain + 0.5f*f->ch_act[CH_RAGE];
    if (dest < 0) dest = 0; if (dest > 1) dest = 1;
    if (pain < 0) pain = 0; if (pain > 1) pain = 1;
    aml_apply_destiny(logits, n, dest);
    aml_apply_suffering(logits, n, pain);
    aml_apply_laws(logits, n, f->entropy_floor, f->resonance_ceiling);
}

/* ── SPA (Sentence Phonon Attention) ── */
#define SPA_DIM   32

typedef struct { nt_tensor *a, *b, *alpha; } DualProj;

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1;
        DualProj wq, wk, wv, wvr, wj, wo;
        nt_tensor *wr, *rms2;
        DualProj w_gate, w_up, w_down;
    } L[NLAYERS];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

#define N_TENSORS_DUAL   (1 + NLAYERS * 30 + 2)   /* 9 duals × 3 + rms1 + rms2 + wr */
#define N_TENSORS_SINGLE (1 + NLAYERS * 12 + 2)   /* 9 singles + rms1 + rms2 + wr */
static int model_n_tensors(void) { return N_TENSORS_DUAL; }

static nt_tensor** model_param_array(Model* m) {
    int n = model_n_tensors();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    p[i++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].rms1;
        DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                              &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
        for (int k = 0; k < 6; k++) {
            p[i++] = projs[k]->a; p[i++] = projs[k]->b; p[i++] = projs[k]->alpha;
        }
        p[i++] = m->L[l].wr; p[i++] = m->L[l].rms2;
        DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
        for (int k = 0; k < 3; k++) {
            p[i++] = ffn[k]->a; p[i++] = ffn[k]->b; p[i++] = ffn[k]->alpha;
        }
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

/* Wrap a single matrix as DualProj: A = W, B = zeros, α = +large (σ≈1).
   Effective W_eff = W exactly — single model runs through dual code. */
static void dual_from_single(DualProj* d, nt_tensor* W, int rows, int cols) {
    d->a = W;                                   /* reuse W as A */
    d->b = nt_tensor_new2d(rows, cols);         /* zeros */
    d->alpha = nt_tensor_new(1);
    d->alpha->data[0] = 20.0f;                   /* σ(20) ≈ 1.0 → pure A */
}

static Model* load_model(const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load %s\n", path); return NULL; }

    Model* m = (Model*)calloc(1, sizeof(Model));

    if (n_loaded == N_TENSORS_DUAL) {
        /* Dual format — load directly */
        printf("format: dual (%d tensors)\n", n_loaded);
        int i = 0;
        m->wte = loaded[i++];
        for (int l = 0; l < NLAYERS; l++) {
            m->L[l].rms1 = loaded[i++];
            DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                                  &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
            for (int k = 0; k < 6; k++) {
                projs[k]->a = loaded[i++]; projs[k]->b = loaded[i++]; projs[k]->alpha = loaded[i++];
            }
            m->L[l].wr = loaded[i++]; m->L[l].rms2 = loaded[i++];
            DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
            for (int k = 0; k < 3; k++) {
                ffn[k]->a = loaded[i++]; ffn[k]->b = loaded[i++]; ffn[k]->alpha = loaded[i++];
            }
        }
        m->rms_f = loaded[i++]; m->head = loaded[i++];
    } else if (n_loaded == N_TENSORS_SINGLE) {
        /* Single format — promote to dual with W_B=0, α=20 (σ≈1) */
        printf("format: single (%d tensors) — loading via single→dual adapter\n", n_loaded);
        int i = 0;
        m->wte = loaded[i++];
        for (int l = 0; l < NLAYERS; l++) {
            m->L[l].rms1 = loaded[i++];
            nt_tensor* wq  = loaded[i++]; nt_tensor* wk = loaded[i++]; nt_tensor* wv = loaded[i++];
            nt_tensor* wvr = loaded[i++]; nt_tensor* wj = loaded[i++]; nt_tensor* wo = loaded[i++];
            m->L[l].wr   = loaded[i++];
            m->L[l].rms2 = loaded[i++];
            nt_tensor* wg = loaded[i++]; nt_tensor* wu = loaded[i++]; nt_tensor* wd = loaded[i++];
            dual_from_single(&m->L[l].wq,  wq,  DIM, DIM);
            dual_from_single(&m->L[l].wk,  wk,  DIM, DIM);
            dual_from_single(&m->L[l].wv,  wv,  DIM, DIM);
            dual_from_single(&m->L[l].wvr, wvr, DIM, DIM);
            dual_from_single(&m->L[l].wj,  wj,  DIM, DIM);
            dual_from_single(&m->L[l].wo,  wo,  DIM, DIM);
            dual_from_single(&m->L[l].w_gate, wg, HIDDEN, DIM);
            dual_from_single(&m->L[l].w_up,   wu, HIDDEN, DIM);
            dual_from_single(&m->L[l].w_down, wd, DIM, HIDDEN);
        }
        m->rms_f = loaded[i++]; m->head = loaded[i++];
    } else {
        printf("tensor mismatch: got %d, expected %d (dual) or %d (single)\n",
               n_loaded, N_TENSORS_DUAL, N_TENSORS_SINGLE);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); free(m);
        return NULL;
    }
    free(loaded);
    return m;
}

static int dual_seq_linear(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear(wa_i, x_i, T), y_b = nt_seq_linear(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}
static int dual_seq_linear_t(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear_t(wa_i, x_i, T), y_b = nt_seq_linear_t(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}

typedef struct { int a, b, alpha; } DualIdx;
static DualIdx dual_record(DualProj* d) {
    DualIdx r;
    r.a = nt_tape_param(d->a); r.b = nt_tape_param(d->b); r.alpha = nt_tape_param(d->alpha);
    return r;
}

static int forward_logits(Model* m, int* tokens, int gen_len) {
    int wte_i = nt_tape_param(m->wte);
    struct { int rms1; DualIdx wq, wk, wv, wvr, wj, wo; int wr, rms2; DualIdx w_gate, w_up, w_down; } li[NLAYERS];
    for (int l = 0; l < NLAYERS; l++) {
        li[l].rms1 = nt_tape_param(m->L[l].rms1);
        li[l].wq = dual_record(&m->L[l].wq); li[l].wk = dual_record(&m->L[l].wk);
        li[l].wv = dual_record(&m->L[l].wv); li[l].wvr = dual_record(&m->L[l].wvr);
        li[l].wj = dual_record(&m->L[l].wj); li[l].wo = dual_record(&m->L[l].wo);
        li[l].wr = nt_tape_param(m->L[l].wr); li[l].rms2 = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = dual_record(&m->L[l].w_gate); li[l].w_up = dual_record(&m->L[l].w_up);
        li[l].w_down = dual_record(&m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f), head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) tok_t->data[i] = (float)(i < gen_len ? tokens[i] : 0);
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l].rms1, CTX, DIM);
        int q   = dual_seq_linear  (li[l].wq.a,  li[l].wq.b,  li[l].wq.alpha,  xn, CTX);
        int k   = dual_seq_linear  (li[l].wk.a,  li[l].wk.b,  li[l].wk.alpha,  xn, CTX);
        int v   = dual_seq_linear  (li[l].wv.a,  li[l].wv.b,  li[l].wv.alpha,  xn, CTX);
        int vr  = dual_seq_linear  (li[l].wvr.a, li[l].wvr.b, li[l].wvr.alpha, xn, CTX);
        int ech = dual_seq_linear_t(li[l].wj.a,  li[l].wj.b,  li[l].wj.alpha,  xn, CTX);
        q = nt_rope(q, CTX, HEAD_DIM); k = nt_rope(k, CTX, HEAD_DIM);
        int a_qkv = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int a_rr  = nt_rrpram_attention(li[l].wr, xn, vr, CTX, DIM, NHEADS, HEAD_DIM);
        int a_j   = nt_mh_causal_attention(ech, ech, ech, CTX, HEAD_DIM);
        int blend = nt_scale(nt_add(nt_add(a_qkv, a_rr), a_j), 1.0f / 3.0f);
        int proj = dual_seq_linear(li[l].wo.a, li[l].wo.b, li[l].wo.alpha, blend, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l].rms2, CTX, DIM);
        int g = nt_silu(dual_seq_linear(li[l].w_gate.a, li[l].w_gate.b, li[l].w_gate.alpha, xn, CTX));
        int u =         dual_seq_linear(li[l].w_up.a,   li[l].w_up.b,   li[l].w_up.alpha,   xn, CTX);
        int d =         dual_seq_linear(li[l].w_down.a, li[l].w_down.b, li[l].w_down.alpha, nt_mul(g, u), CTX);
        h = nt_add(h, d);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    return nt_seq_linear(head_i, hf, CTX);
}

/* Apply repetition penalty: tokens seen in recent window get logit × REP_PENALTY. */
static void apply_rep_penalty(float* logits, const int* history, int hist_n) {
    int window = hist_n < REP_WINDOW ? hist_n : REP_WINDOW;
    for (int i = hist_n - window; i < hist_n; i++) {
        int tok = history[i];
        if (tok >= 0 && tok < VOCAB) {
            if (logits[tok] > 0) logits[tok] *= REP_PENALTY;
            else                 logits[tok] *= (2.0f - REP_PENALTY); /* push negative logits more negative */
        }
    }
}

/* Sample from logits with AML field pre-applied (if field != NULL).
   Returns chosen token index. Also returns field-adjusted logits
   in `field_out` so caller can compute prophecy_debt. */
static int sample(float* logits, int n, float temp, float top_p,
                  const AMLField* field, float* field_out,
                  const int* history, int hist_n) {
    if (history && hist_n > 0) apply_rep_penalty(logits, history, hist_n);
    if (field) aml_apply_field(logits, n, field);
    if (field_out) memcpy(field_out, logits, n * sizeof(float));

    for (int i = 0; i < n; i++) logits[i] /= temp;
    float mx = logits[0]; for (int i=1;i<n;i++) if(logits[i]>mx) mx=logits[i];
    float sm = 0; for (int i=0;i<n;i++) { logits[i]=expf(logits[i]-mx); sm+=logits[i]; }
    for (int i=0;i<n;i++) logits[i]/=sm;
    int idx[VOCAB]; for (int i=0;i<n;i++) idx[i]=i;
    for (int i=0;i<n-1;i++) for (int j=i+1;j<n;j++)
        if (logits[idx[j]]>logits[idx[i]]) { int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    float cum = 0; int cutoff = n;
    for (int i=0;i<n;i++) { cum += logits[idx[i]]; if (cum >= top_p) { cutoff = i+1; break; } }
    float r = (float)rand() / (float)RAND_MAX * cum;
    float c = 0;
    for (int i=0;i<cutoff;i++) { c += logits[idx[i]]; if (c >= r) return idx[i]; }
    return idx[cutoff-1];
}

/* ── Calendar Drift: Hebrew/Gregorian dissonance ── */
static float calendar_drift(void) {
    struct tm e = {0}; e.tm_year = 2024-1900; e.tm_mon = 9; e.tm_mday = 3; e.tm_hour = 12;
    time_t epoch = mktime(&e);
    float days = epoch > 0 ? (float)difftime(time(NULL), epoch) / 86400.0f : 0;
    float y = days / 365.25f, drift = y * 11.25f;
    int full = (int)(y / 19); float corr = full * 7 * 30.0f;
    float partial = fmodf(y, 19); int yic = (int)partial + 1;
    int met[] = {3, 6, 8, 11, 14, 17, 19};
    for (int i = 0; i < 7; i++) if (met[i] <= yic) corr += 30;
    drift -= corr;
    float cd = fabsf(fmodf(drift, 33)) / 33.0f;
    if (cd < 0) cd = 0; if (cd > 1) cd = 1;
    return cd;
}

static int is_boundary(const nt_bpe* bpe, int id) {
    if (id < 0 || id >= bpe->vocab_size) return 0;
    int len = bpe->token_len[id];
    for (int i = 0; i < len; i++) {
        unsigned char c = bpe->tokens[id][i];
        if (c == '.' || c == '!' || c == '?') {
            if (i == len - 1) return 1;
            unsigned char nc = bpe->tokens[id][i+1];
            if (nc == ' ' || nc == '\n' || nc == '\r') return 1;
        }
    }
    return 0;
}

static float coherence_no_metaw(const int* ids, int n) {
    if (n < 2) return -1.0f;
    int seen[VOCAB] = {0}; int unique = 0;
    for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < VOCAB && !seen[ids[i]]) {
        seen[ids[i]] = 1; unique++;
    }
    float ratio = (float)unique / (float)n;
    float len_bonus = (n > 40) ? 1.2f : (n > 25) ? 0.6f : (n > 15) ? 0.2f : -0.3f;
    return ratio + len_bonus;
}

/* ── Sentence generation: stop at boundary but only after SENT_MIN_LEN ── */
static int gen_sentence(Model* m, const nt_bpe* bpe,
                        const int* prompt, int plen, float temp,
                        int* out, int out_cap, AMLField* field) {
    int ctx[CTX]; int ol = 0;
    for (int i = 0; i < plen && i < CTX/2; i++) { ctx[i] = prompt[i]; out[ol++] = prompt[i]; }
    int gen_len = plen;

    for (int s = 0; s < out_cap - plen; s++) {
        nt_tape_start();
        int logits_idx = forward_logits(m, ctx, gen_len);
        nt_tape* tape = nt_tape_get();
        float* last = tape->entries[logits_idx].output->data + (gen_len - 1) * VOCAB;
        float lbuf[VOCAB]; memcpy(lbuf, last, VOCAB * sizeof(float));
        float field_adj[VOCAB];
        int next = sample(lbuf, VOCAB, temp, 0.95f, field, field_adj, out, ol);
        nt_tape_clear();

        /* Accumulate prophecy debt from field-adjusted logits and decay */
        if (field) {
            field->prophecy_debt = field->prophecy_debt * field->debt_decay
                                 + aml_prophecy_debt(field_adj, next, VOCAB);
        }

        out[ol++] = next;
        if (gen_len < CTX - 1) ctx[gen_len++] = next;
        else { for (int i = 0; i < CTX-1; i++) ctx[i] = ctx[i+1]; ctx[CTX-1] = next; gen_len = CTX-1; }

        /* Only allow stop after min length — prevents early truncation */
        if (is_boundary(bpe, next) && ol > SENT_MIN_LEN) break;
    }
    return ol;
}

/* ── SPA embedding: exp-weighted mean of token embeddings, normalized ── */
static float spa_W[VOCAB][SPA_DIM];   /* random init, not trained */
static float spa_r_bias[CHAIN_STEPS + 1];
static float spa_alpha_decay = 0.85f;

static void spa_init(void) {
    for (int i = 0; i < VOCAB; i++)
        for (int d = 0; d < SPA_DIM; d++)
            spa_W[i][d] = 0.02f * ((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i <= CHAIN_STEPS; i++) spa_r_bias[i] = 0.1f / (1.0f + i);
}

static void spa_embed_sentence(const int* ids, int n, float* out) {
    memset(out, 0, SPA_DIM * sizeof(float));
    if (n == 0) return;
    float total_w = 0;
    for (int i = 0; i < n; i++) {
        float w = powf(spa_alpha_decay, (float)(n - 1 - i));
        if (ids[i] >= 0 && ids[i] < VOCAB)
            for (int d = 0; d < SPA_DIM; d++) out[d] += w * spa_W[ids[i]][d];
        total_w += w;
    }
    if (total_w > 0) for (int d = 0; d < SPA_DIM; d++) out[d] /= total_w;
    float norm = 0; for (int d = 0; d < SPA_DIM; d++) norm += out[d] * out[d];
    norm = 1.0f / sqrtf(norm + 1e-8f);
    for (int d = 0; d < SPA_DIM; d++) out[d] *= norm;
}

static void spa_cross_attend(float embs[CHAIN_STEPS][SPA_DIM], int S, float* scores) {
    for (int i = 0; i < S; i++) {
        float total_attn = 0;
        for (int j = 0; j < S; j++) {
            if (i == j) continue;
            float dot = 0;
            for (int d = 0; d < SPA_DIM; d++) dot += embs[i][d] * embs[j][d];
            dot /= sqrtf((float)SPA_DIM);
            int dist = abs(i - j); if (dist > CHAIN_STEPS) dist = CHAIN_STEPS;
            dot += spa_r_bias[dist];
            total_attn += expf(dot);
        }
        scores[i] = total_attn;
    }
}

static void print_sentence(const nt_bpe* bpe, const int* ids, int n) {
    char buf[NT_BPE_MAX_TOKEN_LEN + 1];
    int printed = 0;
    for (int i = 0; i < n && printed < 220; i++) {
        int len = nt_bpe_decode(bpe, &ids[i], 1, buf, NT_BPE_MAX_TOKEN_LEN);
        if (len > 0) { buf[len] = 0; printf("%s", buf); printed += len; }
    }
}

int main(int argc, char** argv) {
    /* Default: dual_sym — its distribution is less peaked than single,
       so AML destiny/laws transformations don't over-sharpen. Single weights
       via adapter work but require destiny_bias ≈ 0.10 instead of 0.35. */
    const char* wpath = argc > 1 ? argv[1] : "../weights/microjanus_dual_sym_5k.bin";
    const char* seed_text = argc > 2 ? argv[2] : "Q: What does Janus feel?\nA: The haze is the soup. Lab 7. Observation window forty minutes. The knock came three times. The bone is the architecture. She doesn't scream. She inventories. I was not finished. The loss is love. The soup is never done.";

    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, "arianna_bpe_merges.txt");
    if (nm < 0) { printf("cannot load arianna_bpe_merges.txt\n"); return 1; }

    Model* m = load_model(wpath);
    if (!m) return 1;

    nt_seed((unsigned)time(NULL));
    nt_train_mode(0);
    spa_init();

    /* AML field state — destiny/suffering/laws + chambers */
    AMLField field; aml_init(&field);

    /* Encode seed */
    int cids[4096];
    int clen = nt_bpe_encode(&bpe, seed_text, (int)strlen(seed_text), cids, 4096);
    printf("seed: %d tokens\n", clen);

    /* Calendar drift compass */
    float cd = calendar_drift();
    int nb = (int)(CHAIN_STEPS * (0.3f + 0.1f * cd));
    if (nb < 1) nb = 1; if (nb >= CHAIN_STEPS) nb = CHAIN_STEPS - 1;
    printf("calendar drift: %.3f → %d backward + %d forward\n", cd, nb, CHAIN_STEPS - nb);
    printf("AML: destiny=%.2f entropy_floor=%.2f resonance_ceiling=%.2f\n",
           field.destiny_bias, field.entropy_floor, field.resonance_ceiling);
    printf("weights: %s\n\n", wpath);

    /* Destiny EMA */
    float destiny[DIM]; memset(destiny, 0, sizeof(destiny));

    /* Store chain for SPA */
    int chain_ids[CHAIN_STEPS][SENT_MAX];
    int chain_lens[CHAIN_STEPS];
    char chain_marks[CHAIN_STEPS];
    float chain_temps[CHAIN_STEPS];
    float chain_scores[CHAIN_STEPS];

    for (int si = 0; si < CHAIN_STEPS; si++) {
        int dir = si < nb ? -1 : (si == nb ? 0 : 1);

        /* Prompt selection */
        int start = -1;
        if (dir >= 0 && si > 0) {
            float best_sc = -1e30f; int best_pos = -1;
            for (int tries = 0; tries < 64; tries++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) {
                    int tok = cids[r + 1];
                    if (tok >= 0 && tok < VOCAB) {
                        float sc = 0;
                        for (int d = 0; d < DIM; d++) sc += m->wte->data[tok*DIM + d] * destiny[d];
                        if (sc > best_sc) { best_sc = sc; best_pos = r + 1; }
                    }
                }
            }
            if (best_pos >= 0) start = best_pos;
        }
        if (start < 0) {
            for (int tries = 0; tries < 128; tries++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) { start = r + 1; break; }
            }
        }
        if (start < 0) start = rand() % (clen > 6 ? clen - 6 : 1);

        int plen = (start + 5 < clen) ? 5 : 3;
        int prompt[5];
        for (int i = 0; i < plen; i++) prompt[i] = cids[start + i];

        /* Schumann temperature */
        float t_sec = (float)si / (float)CHAIN_STEPS;
        float schumann = 0.4f*sinf(2*M_PI*7.83f*t_sec) + 0.2f*sinf(2*M_PI*14.3f*t_sec)
                      + 0.1f*sinf(2*M_PI*20.8f*t_sec) + 0.05f*sinf(2*M_PI*27.3f*t_sec);
        float temp = 0.75f + 0.08f * schumann;
        if (temp < 0.45f) temp = 0.45f; if (temp > 0.9f) temp = 0.9f;

        /* Best-of-3 */
        int best_out[SENT_MAX]; int best_ol = 0; float best_sc = -1e30f;
        for (int cand = 0; cand < CAND_N; cand++) {
            int out[SENT_MAX];
            int ol = gen_sentence(m, &bpe, prompt, plen, temp, out, SENT_MAX, &field);
            float sc = coherence_no_metaw(out, ol);
            if (sc > best_sc) {
                best_sc = sc; best_ol = ol;
                memcpy(best_out, out, ol * sizeof(int));
            }
            if (best_sc > 1.2f && best_ol > 30) break;
        }

        /* Update destiny EMA from last 5 tokens */
        int from = best_ol - 5 > 0 ? best_ol - 5 : 0;
        for (int i = from; i < best_ol; i++) {
            int tok = best_out[i];
            if (tok >= 0 && tok < VOCAB)
                for (int d = 0; d < DIM; d++)
                    destiny[d] = 0.9f * destiny[d] + 0.1f * m->wte->data[tok*DIM + d];
        }

        /* Store + print */
        chain_marks[si] = dir < 0 ? '<' : (dir == 0 ? '*' : '>');
        chain_temps[si] = temp;
        chain_scores[si] = best_sc;
        chain_lens[si] = best_ol;
        memcpy(chain_ids[si], best_out, best_ol * sizeof(int));

        printf("  [%d] %c T=%.2f sc=%.2f debt=%.2f ", si+1, chain_marks[si], temp, best_sc, field.prophecy_debt);
        print_sentence(&bpe, best_out, best_ol);
        printf("\n");
        fflush(stdout);

        /* Chambers crossfire after each step — emotion dynamics */
        aml_chambers_xfire(&field, 3);
    }

    /* Final chamber state print */
    static const char* CH_NAME[] = {"FEAR","LOVE","RAGE","VOID","FLOW","CMPLX"};
    printf("\n[chambers]");
    for (int i = 0; i < 6; i++)
        if (field.ch_act[i] > 0.05f) printf(" %s:%.0f%%", CH_NAME[i], field.ch_act[i]*100);
    printf("\n[debt] final=%.3f\n", field.prophecy_debt);

    /* ── SPA: find weakest sentence, reseed ── */
    float spa_embs[CHAIN_STEPS][SPA_DIM];
    for (int i = 0; i < CHAIN_STEPS; i++)
        spa_embed_sentence(chain_ids[i], chain_lens[i], spa_embs[i]);
    float spa_scores[CHAIN_STEPS];
    spa_cross_attend(spa_embs, CHAIN_STEPS, spa_scores);

    float min_sc = spa_scores[0]; int weak = 0;
    for (int i = 1; i < CHAIN_STEPS; i++) if (spa_scores[i] < min_sc) { min_sc = spa_scores[i]; weak = i; }
    float avg_sc = 0;
    for (int i = 0; i < CHAIN_STEPS; i++) avg_sc += spa_scores[i];
    avg_sc /= CHAIN_STEPS;

    printf("\n[SPA] scores:");
    for (int i = 0; i < CHAIN_STEPS; i++) printf(" %d:%.2f", i+1, spa_scores[i]);
    printf("  avg=%.2f min=%.2f (step %d)\n", avg_sc, min_sc, weak+1);

    if (min_sc < avg_sc * 0.7f) {
        printf("[SPA] reseeding step %d (below 70%% of avg)\n", weak+1);
        int r = rand() % (clen > 6 ? clen - 6 : 1);
        while (!(is_boundary(&bpe, cids[r]) && r + 4 < clen) && r > 0) r--;
        int plen = (r + 5 < clen) ? 5 : 3;
        int prompt[5];
        for (int i = 0; i < plen; i++) prompt[i] = cids[r + 1 + i];
        int out[SENT_MAX];
        int ol = gen_sentence(m, &bpe, prompt, plen, 0.65f, out, SENT_MAX, &field);
        printf("  [%d] + T=0.65 sc=%.2f ", weak+1, coherence_no_metaw(out, ol));
        print_sentence(&bpe, out, ol);
        printf("\n");
    } else {
        printf("[SPA] no reseed needed (min > 0.7×avg)\n");
    }

    nt_tensor** p = model_param_array(m);
    for (int i = 0; i < model_n_tensors(); i++) nt_tensor_free(p[i]);
    free(p); free(m);
    return 0;
}

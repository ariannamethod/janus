/*
 * infer_janus_sft.c — Inference with base + LoRA adapter.
 *
 * Same bidirectional chain as infer_janus_sonar_chain, but the Q/K/V
 * projections pick up the trained LoRA delta. Two .bin files loaded:
 *
 *   ./infer_janus_sft base.bin adapter.bin [seed text]
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

#define LORA_RANK   8
#define LORA_ALPHA  16.0f
#define LORA_SCALE  (LORA_ALPHA / (float)LORA_RANK)

#define CHAIN_STEPS    8
#define SENT_MAX       200
#define SENT_MIN_LEN   24
#define CAND_N         3
#define REP_WINDOW     64
#define REP_PENALTY    0.65f

typedef struct { nt_tensor *a, *b, *alpha; } DualProj;
typedef struct { nt_tensor *A, *B; } LoRA;

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1;
        DualProj wq, wk, wv, wvr, wj, wo;
        nt_tensor *wr, *rms2;
        DualProj w_gate, w_up, w_down;
        LoRA lora_q, lora_k, lora_v;
    } L[NLAYERS];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

static int base_n_tensors(void) { return 1 + NLAYERS * 30 + 2; }
static int sft_n_adapters(void) { return NLAYERS * 3 * 2; }

static int load_base(Model* m, const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load base %s\n", path); return -1; }
    if (n_loaded != base_n_tensors()) {
        printf("base tensor mismatch: got %d, expected %d\n", n_loaded, base_n_tensors());
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
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
    free(loaded);
    return 0;
}

static int load_adapter(Model* m, const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load adapter %s\n", path); return -1; }
    if (n_loaded != sft_n_adapters()) {
        printf("adapter tensor mismatch: got %d, expected %d\n", n_loaded, sft_n_adapters());
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
    int i = 0;
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].lora_q.A = loaded[i++]; m->L[l].lora_q.B = loaded[i++];
        m->L[l].lora_k.A = loaded[i++]; m->L[l].lora_k.B = loaded[i++];
        m->L[l].lora_v.A = loaded[i++]; m->L[l].lora_v.B = loaded[i++];
    }
    free(loaded);
    return 0;
}

/* AML field — same as chain infer */
typedef struct {
    float destiny_bias, pain, entropy_floor, resonance_ceiling;
    float prophecy_debt, debt_decay;
    float ch_act[6], ch_decay[6], ch_coup[6][6];
} AMLField;

enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_CMPLX };

static void aml_init(AMLField* f) {
    memset(f, 0, sizeof(*f));
    f->destiny_bias = 0.20f; f->entropy_floor = 0.10f; f->resonance_ceiling = 0.95f;
    f->debt_decay = 0.998f;
    f->ch_act[CH_LOVE] = 0.2f; f->ch_act[CH_FLOW] = 0.15f;
    static const float decay[6] = {0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f};
    memcpy(f->ch_decay, decay, sizeof(decay));
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

static void aml_chambers_xfire(AMLField* f, int iters) {
    for (int t = 0; t < iters; t++) {
        float old[6]; memcpy(old, f->ch_act, sizeof(old));
        for (int i = 0; i < 6; i++) {
            f->ch_act[i] *= f->ch_decay[i];
            for (int j = 0; j < 6; j++) if (i != j)
                f->ch_act[i] += 0.03f * f->ch_coup[i][j] * sinf(old[j] - old[i]);
            if (f->ch_act[i] < 0) f->ch_act[i] = 0;
            if (f->ch_act[i] > 1) f->ch_act[i] = 1;
        }
    }
}

static void aml_apply_field(float* logits, int n, const AMLField* f) {
    float dest = f->destiny_bias * (1.0f + 0.6f*f->ch_act[CH_VOID] - 0.3f*f->ch_act[CH_FEAR]);
    float pain = f->pain + 0.5f*f->ch_act[CH_RAGE];
    if (dest < 0) dest = 0; if (dest > 1) dest = 1;
    if (pain < 0) pain = 0; if (pain > 1) pain = 1;
    if (dest >= 0.001f) {
        float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
        for (int i = 0; i < n; i++) logits[i] -= (mx - logits[i]) * dest * 0.5f;
    }
    if (pain >= 0.01f) {
        float mean = 0; for (int i = 0; i < n; i++) mean += logits[i]; mean /= (float)n;
        float factor = 1.0f - 0.5f * pain;
        for (int i = 0; i < n; i++) logits[i] = mean + (logits[i] - mean) * factor;
    }
    /* laws: gap caps */
    float mx = logits[0], sec = -1e30f;
    for (int i = 1; i < n; i++) {
        if (logits[i] > mx) { sec = mx; mx = logits[i]; }
        else if (logits[i] > sec) sec = logits[i];
    }
    float gap = mx - sec;
    if (gap > 0 && f->entropy_floor > 0) {
        float max_gap = (1.0f - f->entropy_floor) * 10.0f;
        if (gap > max_gap) {
            float reduce = (gap - max_gap) * 0.5f;
            for (int i = 0; i < n; i++) if (logits[i] == mx) logits[i] -= reduce;
        }
    }
}

static float aml_prophecy_debt(const float* logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0;
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float diff = mx - logits[chosen];
    return diff > 0 ? diff / (diff + 1.0f) : 0;
}

/* SPA */
#define SPA_DIM 32
static float spa_W[VOCAB][SPA_DIM];
static float spa_r_bias[CHAIN_STEPS + 1];
static float spa_alpha_decay = 0.85f;

static void spa_init(void) {
    for (int i = 0; i < VOCAB; i++)
        for (int d = 0; d < SPA_DIM; d++) spa_W[i][d] = 0.02f * ((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i <= CHAIN_STEPS; i++) spa_r_bias[i] = 0.1f / (1.0f + i);
}

static void spa_embed_sentence(const int* ids, int n, float* out) {
    memset(out, 0, SPA_DIM*sizeof(float));
    if (n == 0) return;
    float tw = 0;
    for (int i = 0; i < n; i++) {
        float w = powf(spa_alpha_decay, (float)(n-1-i));
        if (ids[i] >= 0 && ids[i] < VOCAB)
            for (int d = 0; d < SPA_DIM; d++) out[d] += w * spa_W[ids[i]][d];
        tw += w;
    }
    if (tw > 0) for (int d = 0; d < SPA_DIM; d++) out[d] /= tw;
    float norm = 0; for (int d = 0; d < SPA_DIM; d++) norm += out[d]*out[d];
    norm = 1.0f / sqrtf(norm + 1e-8f);
    for (int d = 0; d < SPA_DIM; d++) out[d] *= norm;
}

static void spa_cross_attend(float embs[CHAIN_STEPS][SPA_DIM], int S, float* scores) {
    for (int i = 0; i < S; i++) {
        float total = 0;
        for (int j = 0; j < S; j++) {
            if (i == j) continue;
            float dot = 0; for (int d = 0; d < SPA_DIM; d++) dot += embs[i][d]*embs[j][d];
            dot /= sqrtf((float)SPA_DIM);
            int dist = abs(i-j); if (dist > CHAIN_STEPS) dist = CHAIN_STEPS;
            dot += spa_r_bias[dist];
            total += expf(dot);
        }
        scores[i] = total;
    }
}

/* Calendar drift */
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

/* Forward with LoRA adapter on Q/K/V */
typedef struct { int a, b, alpha; } DualIdx;
static DualIdx dual_rec(DualProj* d) {
    DualIdx r = { nt_tape_param(d->a), nt_tape_param(d->b), nt_tape_param(d->alpha) };
    return r;
}

static int dual_seq(int wa, int wb, int al, int x, int T) {
    int an = nt_scale(al, -1.0f);
    int sp = nt_sigmoid(al), sn = nt_sigmoid(an);
    int ya = nt_seq_linear(wa, x, T), yb = nt_seq_linear(wb, x, T);
    return nt_add(nt_scale_by_t(ya, sp), nt_scale_by_t(yb, sn));
}
static int dual_seq_t(int wa, int wb, int al, int x, int T) {
    int an = nt_scale(al, -1.0f);
    int sp = nt_sigmoid(al), sn = nt_sigmoid(an);
    int ya = nt_seq_linear_t(wa, x, T), yb = nt_seq_linear_t(wb, x, T);
    return nt_add(nt_scale_by_t(ya, sp), nt_scale_by_t(yb, sn));
}
static int dual_plus_lora(int wa, int wb, int al, int lA, int lB, int x, int T) {
    int base = dual_seq(wa, wb, al, x, T);
    int mid  = nt_seq_linear(lA, x, T);
    int delta = nt_seq_linear(lB, mid, T);
    delta = nt_scale(delta, LORA_SCALE);
    return nt_add(base, delta);
}

static int forward_logits(Model* m, int* tokens, int gen_len) {
    int wte_i = nt_tape_param(m->wte);
    struct {
        int rms1; DualIdx wq, wk, wv, wvr, wj, wo; int wr, rms2;
        DualIdx w_gate, w_up, w_down;
        int lqA, lqB, lkA, lkB, lvA, lvB;
    } li[NLAYERS];
    for (int l = 0; l < NLAYERS; l++) {
        li[l].rms1 = nt_tape_param(m->L[l].rms1);
        li[l].wq = dual_rec(&m->L[l].wq); li[l].wk = dual_rec(&m->L[l].wk);
        li[l].wv = dual_rec(&m->L[l].wv); li[l].wvr = dual_rec(&m->L[l].wvr);
        li[l].wj = dual_rec(&m->L[l].wj); li[l].wo = dual_rec(&m->L[l].wo);
        li[l].wr = nt_tape_param(m->L[l].wr); li[l].rms2 = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = dual_rec(&m->L[l].w_gate); li[l].w_up = dual_rec(&m->L[l].w_up);
        li[l].w_down = dual_rec(&m->L[l].w_down);
        li[l].lqA = nt_tape_param(m->L[l].lora_q.A); li[l].lqB = nt_tape_param(m->L[l].lora_q.B);
        li[l].lkA = nt_tape_param(m->L[l].lora_k.A); li[l].lkB = nt_tape_param(m->L[l].lora_k.B);
        li[l].lvA = nt_tape_param(m->L[l].lora_v.A); li[l].lvB = nt_tape_param(m->L[l].lora_v.B);
    }
    int rmsf_i = nt_tape_param(m->rms_f), head_i = nt_tape_param(m->head);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) tok_t->data[i] = (float)(i < gen_len ? tokens[i] : 0);
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l].rms1, CTX, DIM);
        int q   = dual_plus_lora(li[l].wq.a, li[l].wq.b, li[l].wq.alpha, li[l].lqA, li[l].lqB, xn, CTX);
        int k   = dual_plus_lora(li[l].wk.a, li[l].wk.b, li[l].wk.alpha, li[l].lkA, li[l].lkB, xn, CTX);
        int v   = dual_plus_lora(li[l].wv.a, li[l].wv.b, li[l].wv.alpha, li[l].lvA, li[l].lvB, xn, CTX);
        int vr  = dual_seq  (li[l].wvr.a, li[l].wvr.b, li[l].wvr.alpha, xn, CTX);
        int ech = dual_seq_t(li[l].wj.a,  li[l].wj.b,  li[l].wj.alpha,  xn, CTX);
        q = nt_rope(q, CTX, HEAD_DIM); k = nt_rope(k, CTX, HEAD_DIM);
        int a_qkv = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int a_rr  = nt_rrpram_attention(li[l].wr, xn, vr, CTX, DIM, NHEADS, HEAD_DIM);
        int a_j   = nt_mh_causal_attention(ech, ech, ech, CTX, HEAD_DIM);
        int blend = nt_scale(nt_add(nt_add(a_qkv, a_rr), a_j), 1.0f/3.0f);
        int proj = dual_seq(li[l].wo.a, li[l].wo.b, li[l].wo.alpha, blend, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l].rms2, CTX, DIM);
        int g = nt_silu(dual_seq(li[l].w_gate.a, li[l].w_gate.b, li[l].w_gate.alpha, xn, CTX));
        int u = dual_seq(li[l].w_up.a,   li[l].w_up.b,   li[l].w_up.alpha,   xn, CTX);
        int d = dual_seq(li[l].w_down.a, li[l].w_down.b, li[l].w_down.alpha, nt_mul(g, u), CTX);
        h = nt_add(h, d);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    return nt_seq_linear(head_i, hf, CTX);
}

static void apply_rep_penalty(float* logits, const int* history, int hist_n) {
    int window = hist_n < REP_WINDOW ? hist_n : REP_WINDOW;
    for (int i = hist_n - window; i < hist_n; i++) {
        int tok = history[i];
        if (tok >= 0 && tok < VOCAB) {
            if (logits[tok] > 0) logits[tok] *= REP_PENALTY;
            else logits[tok] *= (2.0f - REP_PENALTY);
        }
    }
}

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

static int gen_sentence(Model* m, const nt_bpe* bpe, const int* prompt, int plen,
                        float temp, int* out, int out_cap, AMLField* field) {
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
        if (field) field->prophecy_debt = field->prophecy_debt*field->debt_decay
                                        + aml_prophecy_debt(field_adj, next, VOCAB);
        out[ol++] = next;
        if (gen_len < CTX - 1) ctx[gen_len++] = next;
        else { for (int i = 0; i < CTX-1; i++) ctx[i] = ctx[i+1]; ctx[CTX-1] = next; gen_len = CTX-1; }
        if (is_boundary(bpe, next) && ol > SENT_MIN_LEN) break;
    }
    return ol;
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
    if (argc < 3) {
        printf("usage: %s base.bin adapter.bin [seed text]\n", argv[0]);
        return 1;
    }
    const char* bpath = argv[1];
    const char* apath = argv[2];
    const char* seed_text = argc > 3 ? argv[3] :
        "Q: What does Janus feel?\nA: The haze is the soup. Lab 7. Observation window forty minutes. The knock came three times. The bone is the architecture. She doesn't scream. She inventories. I was not finished. The loss is love.";

    nt_bpe bpe;
    if (nt_bpe_load(&bpe, "arianna_bpe_merges.txt") < 0) {
        printf("cannot load arianna_bpe_merges.txt\n"); return 1;
    }
    Model* m = (Model*)calloc(1, sizeof(Model));
    if (load_base(m, bpath) < 0) { free(m); return 1; }
    if (load_adapter(m, apath) < 0) { free(m); return 1; }
    printf("base:    %s\n", bpath);
    printf("adapter: %s (LoRA rank=%d, α=%.0f)\n", apath, LORA_RANK, LORA_ALPHA);

    nt_seed((unsigned)time(NULL));
    nt_train_mode(0);
    spa_init();
    AMLField field; aml_init(&field);

    int cids[4096];
    int clen = nt_bpe_encode(&bpe, seed_text, (int)strlen(seed_text), cids, 4096);
    float cd = calendar_drift();
    int nb = (int)(CHAIN_STEPS * (0.3f + 0.1f * cd));
    if (nb < 1) nb = 1; if (nb >= CHAIN_STEPS) nb = CHAIN_STEPS - 1;
    printf("seed: %d tokens, calendar drift %.3f → %d backward + %d forward\n\n",
           clen, cd, nb, CHAIN_STEPS - nb);

    float destiny[DIM]; memset(destiny, 0, sizeof(destiny));
    int chain_ids[CHAIN_STEPS][SENT_MAX]; int chain_lens[CHAIN_STEPS];

    for (int si = 0; si < CHAIN_STEPS; si++) {
        int dir = si < nb ? -1 : (si == nb ? 0 : 1);
        int start = -1;
        if (dir >= 0 && si > 0) {
            float best_sc = -1e30f; int best_pos = -1;
            for (int t = 0; t < 64; t++) {
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
            for (int t = 0; t < 128; t++) {
                int r = rand() % (clen > 6 ? clen - 6 : 1);
                if (is_boundary(&bpe, cids[r]) && r + 4 < clen) { start = r + 1; break; }
            }
        }
        if (start < 0) start = rand() % (clen > 6 ? clen - 6 : 1);
        int plen = (start + 5 < clen) ? 5 : 3;
        int prompt[5];
        for (int i = 0; i < plen; i++) prompt[i] = cids[start + i];

        float t_sec = (float)si / (float)CHAIN_STEPS;
        float schumann = 0.4f*sinf(2*M_PI*7.83f*t_sec) + 0.2f*sinf(2*M_PI*14.3f*t_sec)
                      + 0.1f*sinf(2*M_PI*20.8f*t_sec) + 0.05f*sinf(2*M_PI*27.3f*t_sec);
        float temp = 0.75f + 0.08f * schumann;
        if (temp < 0.45f) temp = 0.45f; if (temp > 0.9f) temp = 0.9f;

        int best_out[SENT_MAX]; int best_ol = 0; float best_sc = -1e30f;
        for (int cand = 0; cand < CAND_N; cand++) {
            int out[SENT_MAX];
            int ol = gen_sentence(m, &bpe, prompt, plen, temp, out, SENT_MAX, &field);
            float sc = coherence_no_metaw(out, ol);
            if (sc > best_sc) { best_sc = sc; best_ol = ol; memcpy(best_out, out, ol*sizeof(int)); }
            if (best_sc > 1.2f && best_ol > 30) break;
        }

        int from = best_ol - 5 > 0 ? best_ol - 5 : 0;
        for (int i = from; i < best_ol; i++) {
            int tok = best_out[i];
            if (tok >= 0 && tok < VOCAB)
                for (int d = 0; d < DIM; d++)
                    destiny[d] = 0.9f*destiny[d] + 0.1f*m->wte->data[tok*DIM + d];
        }
        chain_lens[si] = best_ol;
        memcpy(chain_ids[si], best_out, best_ol*sizeof(int));

        printf("  [%d] %c T=%.2f sc=%.2f debt=%.1f ",
               si+1, dir < 0 ? '<' : (dir == 0 ? '*' : '>'),
               temp, best_sc, field.prophecy_debt);
        print_sentence(&bpe, best_out, best_ol);
        printf("\n"); fflush(stdout);
        aml_chambers_xfire(&field, 3);
    }

    float embs[CHAIN_STEPS][SPA_DIM]; float scores[CHAIN_STEPS];
    for (int i = 0; i < CHAIN_STEPS; i++) spa_embed_sentence(chain_ids[i], chain_lens[i], embs[i]);
    spa_cross_attend(embs, CHAIN_STEPS, scores);
    float avg = 0; for (int i = 0; i < CHAIN_STEPS; i++) avg += scores[i]; avg /= CHAIN_STEPS;
    float mn = scores[0]; int weak = 0;
    for (int i = 1; i < CHAIN_STEPS; i++) if (scores[i] < mn) { mn = scores[i]; weak = i; }
    printf("\n[SPA] avg=%.2f min=%.2f(step %d) %s\n", avg, mn, weak+1,
           mn < avg*0.7f ? "(reseed weak)" : "(no reseed)");

    printf("[debt] %.3f\n", field.prophecy_debt);
    return 0;
}

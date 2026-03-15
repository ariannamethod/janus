/*
 * janus-hybrid.c — Janus Hybrid: BPE training + char-level output
 *
 * The architectural pressure: trains on BPE tokens (subword units)
 * but generates output through char-level decoding. This creates
 * a compression/expansion tension — like thinking in concepts
 * but speaking letter by letter.
 *
 * Same hybrid attention (QKV + RRPRAM + Janus self-resonance),
 * same Calendar Drift, AML physics, dual weight matrices.
 * Chuck optimizer. 12 bi-directional associative reasoning steps.
 *
 *   cc janus-hybrid.c -O2 -lm -o janus-hybrid
 *   ./janus-hybrid --train data.txt --steps 5000
 *   ./janus-hybrid --generate "To be or not"
 *
 * By Arianna Method. הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════
 * CONFIGURATION — BPE training, char-level output
 * ═══════════════════════════════════════════════════════════════════ */

#define BPE_VOCAB   512       /* BPE vocab for training */
#define CHAR_VOCAB  256       /* char-level output */
#define MAX_T       256
#define DIM         288
#define HEADS       6
#define HEAD_DIM    (DIM/HEADS)
#define BLOCKS      6
#define MLP_DIM     768
#define MAX_BLK     16
#define NSTEPS      12
#define SENT_LEN    40
#define MAX_MERGES  256       /* BPE merge rules */

/* ═══════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — exact port from ariannamethod.c
 * ═══════════════════════════════════════════════════════════════════ */

#define AM_ANNUAL_DRIFT     11.25f
#define AM_GREGORIAN_YEAR   365.25f
#define AM_METONIC_YEARS    19
#define AM_METONIC_LEAPS    7
#define AM_MAX_UNCORRECTED  33.0f

static const int g_metonic_leap_years[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;

static float clamp01(float x) {
    if (!isfinite(x)) return 0.0f;
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

static void calendar_init(void) {
    struct tm epoch_tm;
    memset(&epoch_tm, 0, sizeof(epoch_tm));
    epoch_tm.tm_year = 2024 - 1900;
    epoch_tm.tm_mon = 10 - 1;
    epoch_tm.tm_mday = 3;
    epoch_tm.tm_hour = 12;
    g_epoch_t = mktime(&epoch_tm);
}

static int calendar_days_since_epoch(void) {
    if (g_epoch_t <= 0) return 0;
    return (int)(difftime(time(NULL), g_epoch_t) / 86400.0);
}

static float calendar_cumulative_drift(int days) {
    float years = (float)days / AM_GREGORIAN_YEAR;
    float base_drift = years * AM_ANNUAL_DRIFT;
    int full_cycles = (int)(years / AM_METONIC_YEARS);
    float corrections = (float)(full_cycles * AM_METONIC_LEAPS) * 30.0f;
    float partial = fmodf(years, (float)AM_METONIC_YEARS);
    int year_in_cycle = (int)partial + 1;
    for (int i = 0; i < AM_METONIC_LEAPS; i++)
        if (g_metonic_leap_years[i] <= year_in_cycle) corrections += 30.0f;
    return base_drift - corrections;
}

static float calendar_dissonance(int days) {
    float drift = calendar_cumulative_drift(days);
    return clamp01(fabsf(fmodf(drift, AM_MAX_UNCORRECTED)) / AM_MAX_UNCORRECTED);
}

/* ═══════════════════════════════════════════════════════════════════
 * METAJANUS — birth snapshot
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int birth_days; float birth_drift, birth_dissonance;
    time_t birth_time; int alive;
} MetaJanus;
static MetaJanus MJ = {0};

static void metajanus_init(void) {
    if (MJ.alive) return;
    calendar_init();
    MJ.birth_days = calendar_days_since_epoch();
    MJ.birth_drift = calendar_cumulative_drift(MJ.birth_days);
    MJ.birth_dissonance = calendar_dissonance(MJ.birth_days);
    MJ.birth_time = time(NULL);
    MJ.alive = 1;
}

static float metajanus_personal_dissonance(void) {
    float now_drift = calendar_cumulative_drift(calendar_days_since_epoch());
    return clamp01(fabsf(now_drift - MJ.birth_drift) / AM_MAX_UNCORRECTED);
}

/* ═══════════════════════════════════════════════════════════════════
 * AML PHYSICS
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float prophecy_debt, destiny_bias, wormhole, resonance;
    float trauma, tension, pain, entropy_floor;
    int prophecy_horizon, tunnel_skip_max;
    float tunnel_threshold;
} AMLState;

static AMLState AML = {
    .prophecy_debt=0, .destiny_bias=0.1f, .wormhole=0.02f, .resonance=0.5f,
    .trauma=0, .tension=0, .pain=0, .entropy_floor=0.01f,
    .prophecy_horizon=12, .tunnel_skip_max=7, .tunnel_threshold=0.55f
};

static float compute_prophecy_debt(const float *logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0.0f;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float d = mx - logits[chosen];
    return d > 0 ? d / (d + 1.0f) : 0.0f;
}

static void apply_destiny(float *logits, int n) {
    if (AML.destiny_bias < 0.001f) return;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) logits[i] -= (mx - logits[i]) * AML.destiny_bias * 0.5f;
}

/* Kuramoto chambers */
enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX, NCH };
static float chambers[NCH] = {0};
static const float ch_decay[NCH] = {0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f};

static void update_chambers(int step_idx) {
    float depth = (float)step_idx / NSTEPS;
    if (depth < 0.33f) chambers[CH_FLOW] += 0.05f;
    else if (depth < 0.66f) chambers[CH_FEAR] += 0.04f;
    else chambers[CH_VOID] += 0.05f;
    if (depth > 0.75f) chambers[CH_COMPLEX] += 0.03f;
    float K = 0.02f, old[NCH];
    memcpy(old, chambers, sizeof(old));
    for (int i = 0; i < NCH; i++) {
        for (int j = 0; j < NCH; j++)
            if (i != j) chambers[i] += K * sinf(old[j] - old[i]);
        chambers[i] = clamp01(chambers[i] * ch_decay[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * BPE TOKENIZER — learns merge rules from data
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int a, b;       /* pair to merge */
    int result;     /* new token ID */
} MergeRule;

typedef struct {
    MergeRule merges[MAX_MERGES];
    int n_merges;
    int vocab_size;  /* 256 base + n_merges */
} BPETokenizer;

static BPETokenizer BPE = { .n_merges = 0, .vocab_size = 256 };

static int bpe_encode(const unsigned char *text, int text_len, int *out, int max_out) {
    /* Start with byte-level tokens */
    int n = 0;
    for (int i = 0; i < text_len && n < max_out; i++)
        out[n++] = text[i];

    /* Apply merge rules greedily */
    for (int m = 0; m < BPE.n_merges; m++) {
        MergeRule *mr = &BPE.merges[m];
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && out[i] == mr->a && out[i+1] == mr->b) {
                out[j++] = mr->result;
                i++;  /* skip next */
            } else {
                out[j++] = out[i];
            }
        }
        n = j;
    }
    return n;
}

static void bpe_learn_merges(const unsigned char *data, int data_len, int n_merges) {
    int *tokens = (int *)malloc(data_len * sizeof(int));
    int n = data_len;
    for (int i = 0; i < n; i++) tokens[i] = data[i];

    for (int m = 0; m < n_merges && m < MAX_MERGES; m++) {
        /* Count pairs */
        int best_a = -1, best_b = -1, best_count = 0;
        int pair_counts[BPE_VOCAB][16];  /* sparse pair counting */
        int pair_ids[BPE_VOCAB][16];
        int pair_n[BPE_VOCAB];
        memset(pair_n, 0, sizeof(pair_n));

        for (int i = 0; i + 1 < n; i++) {
            int a = tokens[i], b = tokens[i+1];
            if (a >= BPE_VOCAB) continue;
            int found = 0;
            for (int j = 0; j < pair_n[a] && j < 16; j++) {
                if (pair_ids[a][j] == b) { pair_counts[a][j]++; found = 1;
                    if (pair_counts[a][j] > best_count) {
                        best_count = pair_counts[a][j]; best_a = a; best_b = b;
                    }
                    break;
                }
            }
            if (!found && pair_n[a] < 16) {
                int idx = pair_n[a]++;
                pair_ids[a][idx] = b;
                pair_counts[a][idx] = 1;
                if (1 > best_count) { best_count = 1; best_a = a; best_b = b; }
            }
        }

        if (best_count < 2) break;

        int new_id = 256 + m;
        BPE.merges[m] = (MergeRule){ best_a, best_b, new_id };
        BPE.n_merges = m + 1;
        BPE.vocab_size = 256 + m + 1;

        /* Apply merge */
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && tokens[i] == best_a && tokens[i+1] == best_b) {
                tokens[j++] = new_id; i++;
            } else {
                tokens[j++] = tokens[i];
            }
        }
        n = j;
    }
    free(tokens);
}

/* ═══════════════════════════════════════════════════════════════════
 * MATH PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════ */

static void matmul(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
}

static void matmul_atb(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[p*m+i] * B[p*n+j];
            C[i*n+j] = s;
        }
}

static void matmul_abt(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
}

static void row_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    if (s > 0) for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20.0f ? x / (1.0f + expf(-x)) : 0.0f; }
static float siluf_grad(float x) {
    if (x < -20.0f) return 0.0f;
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

static void rmsnorm_fwd(float *out, const float *x, const float *g, int T, int E) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int e = 0; e < E; e++) ss += x[t*E+e] * x[t*E+e];
        float inv = 1.0f / sqrtf(ss / E + 1e-5f);
        for (int e = 0; e < E; e++) out[t*E+e] = g[e] * x[t*E+e] * inv;
    }
}

static float randn(void) {
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL — BPE input embedding, char-level output projection
 *
 * Training: BPE tokens → transformer → BPE logits (next-BPE prediction)
 * Generation: BPE context → transformer → char-level output
 *
 * This is THE PRESSURE: conceptual (BPE) thinking, precise (char) output.
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *tok_emb;              /* [BPE_VOCAB, E] — BPE embeddings for training */
    float *pos_emb;              /* [T, E] */
    float *rms1[MAX_BLK];
    float *wq[MAX_BLK], *wk[MAX_BLK], *wv[MAX_BLK];
    float *wr[MAX_BLK], *wvr[MAX_BLK];
    float *wj[MAX_BLK];
    float *gate[MAX_BLK];
    float *wo[MAX_BLK];
    float *rms2[MAX_BLK];
    float *w_gate[MAX_BLK], *w_up[MAX_BLK], *w_down[MAX_BLK];
    float *rms_f;
    float *out_bpe;              /* [E, BPE_VOCAB] — BPE output for training */
    float *out_char;             /* [E, CHAR_VOCAB] — char output for generation */
} Ptrs;

typedef struct {
    int n_params;
    float *data, *grad, *chuck_m, *chuck_v;
    Ptrs w, g;
} Model;

static int model_size(void) {
    int s = BPE_VOCAB * DIM + MAX_T * DIM;
    for (int b = 0; b < BLOCKS; b++) {
        s += DIM;
        s += HEADS * DIM * HEAD_DIM * 3;  /* wq, wk, wv */
        s += HEADS * DIM * MAX_T;          /* wr */
        s += HEADS * DIM * HEAD_DIM;       /* wvr */
        s += DIM * DIM;                    /* wj */
        s += HEADS * 3;                    /* gate */
        s += DIM * DIM;                    /* wo */
        s += DIM;                          /* rms2 */
        s += DIM * MLP_DIM * 2 + MLP_DIM * DIM;  /* SwiGLU */
    }
    s += DIM;
    s += DIM * BPE_VOCAB;    /* BPE output */
    s += DIM * CHAR_VOCAB;   /* char output — THE PRESSURE */
    return s;
}

static void assign_ptrs(Ptrs *p, float *base) {
    float *q = base;
    p->tok_emb = q; q += BPE_VOCAB * DIM;
    p->pos_emb = q; q += MAX_T * DIM;
    for (int b = 0; b < BLOCKS; b++) {
        p->rms1[b]   = q; q += DIM;
        p->wq[b]     = q; q += HEADS * DIM * HEAD_DIM;
        p->wk[b]     = q; q += HEADS * DIM * HEAD_DIM;
        p->wv[b]     = q; q += HEADS * DIM * HEAD_DIM;
        p->wr[b]     = q; q += HEADS * DIM * MAX_T;
        p->wvr[b]    = q; q += HEADS * DIM * HEAD_DIM;
        p->wj[b]     = q; q += DIM * DIM;
        p->gate[b]   = q; q += HEADS * 3;
        p->wo[b]     = q; q += DIM * DIM;
        p->rms2[b]   = q; q += DIM;
        p->w_gate[b] = q; q += DIM * MLP_DIM;
        p->w_up[b]   = q; q += DIM * MLP_DIM;
        p->w_down[b] = q; q += MLP_DIM * DIM;
    }
    p->rms_f    = q; q += DIM;
    p->out_bpe  = q; q += DIM * BPE_VOCAB;
    p->out_char = q; q += DIM * CHAR_VOCAB;
}

static void model_init(Model *m) {
    m->n_params = model_size();
    m->data    = (float *)calloc(m->n_params, sizeof(float));
    m->grad    = (float *)calloc(m->n_params, sizeof(float));
    m->chuck_m = (float *)calloc(m->n_params, sizeof(float));
    m->chuck_v = (float *)calloc(m->n_params, sizeof(float));
    assign_ptrs(&m->w, m->data);
    assign_ptrs(&m->g, m->grad);
    float scale = 0.02f * sqrtf(2.0f / DIM);
    for (int i = 0; i < m->n_params; i++) m->data[i] = randn() * scale;
    for (int b = 0; b < BLOCKS; b++) {
        for (int e = 0; e < DIM; e++) { m->w.rms1[b][e] = 1.0f; m->w.rms2[b][e] = 1.0f; }
    }
    for (int e = 0; e < DIM; e++) m->w.rms_f[e] = 1.0f;
    for (int b = 0; b < BLOCKS; b++)
        for (int i = 0; i < HEADS * 3; i++) m->w.gate[b][i] = 0.0f;
    printf("[janus-hybrid] model: %d params (%.2fMB)\n", m->n_params, m->n_params*4.0f/1e6f);
}

static void model_free(Model *m) {
    free(m->data); free(m->grad); free(m->chuck_m); free(m->chuck_v);
}

/* Dual weight system */
typedef struct { Model A, B; float blend_alpha; float *blended; Ptrs bw; } DualModel;

static void dual_init(DualModel *dm) {
    model_init(&dm->A); model_init(&dm->B);
    dm->blended = (float *)calloc(dm->A.n_params, sizeof(float));
    assign_ptrs(&dm->bw, dm->blended);
    dm->blend_alpha = 0.5f;
}

static void dual_blend(DualModel *dm) {
    float cal_d = calendar_dissonance(calendar_days_since_epoch());
    float meta_d = MJ.alive ? metajanus_personal_dissonance() : 0.5f;
    dm->blend_alpha = clamp01(0.5f + 0.3f*(cal_d-0.5f) - 0.2f*AML.prophecy_debt + 0.1f*meta_d);
    float a = dm->blend_alpha, b = 1.0f - a;
    for (int i = 0; i < dm->A.n_params; i++)
        dm->blended[i] = a * dm->A.data[i] + b * dm->B.data[i];
}

static void dual_free(DualModel *dm) {
    model_free(&dm->A); model_free(&dm->B); free(dm->blended);
}

/* ═══════════════════════════════════════════════════════════════════
 * CHUCK OPTIMIZER
 * ═══════════════════════════════════════════════════════════════════ */

#define CHUCK_B1 0.9f
#define CHUCK_B2 0.999f
#define CHUCK_EPS 1e-8f
#define CHUCK_WINDOW 16

static struct {
    float hist[CHUCK_WINDOW];
    float dampen, noise, sigma, loss_ema, macro_ema, best_macro, lr_scale;
    int macro_stag, pos, full, stag, global_step, step_t;
} Chuck = { .dampen=1, .sigma=1, .lr_scale=1 };

static void chuck_observe(float loss) {
    if (Chuck.loss_ema == 0) Chuck.loss_ema = loss;
    else Chuck.loss_ema = 0.99f * Chuck.loss_ema + 0.01f * loss;
    Chuck.hist[Chuck.pos % CHUCK_WINDOW] = Chuck.loss_ema;
    Chuck.pos++;
    if (Chuck.pos >= CHUCK_WINDOW) Chuck.full = 1;
    if (Chuck.full) {
        int q = CHUCK_WINDOW / 4;
        float recent = 0, old = 0;
        for (int i = 0; i < q; i++) {
            recent += Chuck.hist[(Chuck.pos-1-i) % CHUCK_WINDOW];
            old += Chuck.hist[(Chuck.pos-CHUCK_WINDOW+i) % CHUCK_WINDOW];
        }
        recent /= q; old /= q;
        float trend = (recent - old) / (old + 1e-8f);
        if (trend > 0.01f) Chuck.dampen *= 0.95f;
        if (trend < -0.05f) Chuck.dampen *= 1.05f;
        if (fabsf(trend) < 0.001f) { Chuck.stag++; if (Chuck.stag>8) { Chuck.noise=0.001f; Chuck.stag=0; } }
        else { Chuck.stag = 0; Chuck.noise *= 0.9f; }
        if (Chuck.dampen < 0.3f) Chuck.dampen = 0.3f;
        if (Chuck.dampen > 2.0f) Chuck.dampen = 2.0f;
    }
    Chuck.global_step++;
    if (Chuck.macro_ema == 0) Chuck.macro_ema = loss;
    else Chuck.macro_ema = 0.999f * Chuck.macro_ema + 0.001f * loss;
    if (Chuck.global_step % 500 == 0 && Chuck.global_step > CHUCK_WINDOW) {
        if (Chuck.macro_ema > Chuck.best_macro * 0.999f) {
            Chuck.macro_stag++;
            if (Chuck.macro_stag >= 3) { Chuck.lr_scale *= 0.5f; if (Chuck.lr_scale<0.05f) Chuck.lr_scale=0.05f; Chuck.macro_stag=0; }
        } else { Chuck.best_macro = Chuck.macro_ema; Chuck.macro_stag = 0; }
    }
}

static void chuck_update(float *w, float *g, float *cm, float *cv, int n, float lr) {
    Chuck.step_t++;
    float bc1 = 1.0f - powf(CHUCK_B1, (float)Chuck.step_t);
    float bc2 = 1.0f - powf(CHUCK_B2, (float)Chuck.step_t);
    float eff = lr * Chuck.lr_scale * Chuck.dampen * Chuck.sigma;
    for (int i = 0; i < n; i++) {
        cm[i] = CHUCK_B1*cm[i] + (1-CHUCK_B1)*g[i];
        cv[i] = CHUCK_B2*cv[i] + (1-CHUCK_B2)*g[i]*g[i];
        w[i] -= eff * (cm[i]/bc1) / (sqrtf(cv[i]/bc2) + CHUCK_EPS);
        if (Chuck.noise > 0) w[i] += Chuck.noise * randn() * 0.01f;
        g[i] = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * FORWARD PASS — activations buffer
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *x, *rm1, *cat, *attn_out, *r1, *rm2;
    float *mlp_gate, *mlp_up, *mlp_swi, *mlp_out, *r2;
    float *q, *k, *v, *attn, *head_out;
    float *rrp_attn, *rrp_v, *rrp_out;
    float *j_echo, *j_attn, *j_out;
    float *final_rm, *logits;
} Acts;

static void acts_alloc(Acts *a, int logit_size) {
    a->x = calloc(MAX_T * DIM, sizeof(float));
    a->rm1 = calloc(MAX_T * DIM, sizeof(float));
    a->cat = calloc(MAX_T * DIM, sizeof(float));
    a->attn_out = calloc(MAX_T * DIM, sizeof(float));
    a->r1 = calloc(MAX_T * DIM, sizeof(float));
    a->rm2 = calloc(MAX_T * DIM, sizeof(float));
    a->mlp_gate = calloc(MAX_T * MLP_DIM, sizeof(float));
    a->mlp_up = calloc(MAX_T * MLP_DIM, sizeof(float));
    a->mlp_swi = calloc(MAX_T * MLP_DIM, sizeof(float));
    a->mlp_out = calloc(MAX_T * DIM, sizeof(float));
    a->r2 = calloc(MAX_T * DIM, sizeof(float));
    a->q = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->k = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->v = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->attn = calloc(MAX_T * MAX_T, sizeof(float));
    a->head_out = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->rrp_attn = calloc(MAX_T * MAX_T, sizeof(float));
    a->rrp_v = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->rrp_out = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->j_echo = calloc(MAX_T * DIM, sizeof(float));
    a->j_attn = calloc(MAX_T * MAX_T, sizeof(float));
    a->j_out = calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->final_rm = calloc(MAX_T * DIM, sizeof(float));
    a->logits = calloc(MAX_T * logit_size, sizeof(float));
}

static void acts_free(Acts *a) {
    free(a->x); free(a->rm1); free(a->cat); free(a->attn_out);
    free(a->r1); free(a->rm2); free(a->mlp_gate); free(a->mlp_up);
    free(a->mlp_swi); free(a->mlp_out); free(a->r2);
    free(a->q); free(a->k); free(a->v); free(a->attn); free(a->head_out);
    free(a->rrp_attn); free(a->rrp_v); free(a->rrp_out);
    free(a->j_echo); free(a->j_attn); free(a->j_out);
    free(a->final_rm); free(a->logits);
}

/* Janus self-resonance attention */
static void janus_attention(const float *x, const float *wj, float *echo_out,
                            float *attn_out, int T) {
    float cal_mod = 1.0f + 0.5f * calendar_dissonance(calendar_days_since_epoch());
    float debt_temp = 1.0f + AML.prophecy_debt;
    float *scores = calloc(T, sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *xt = x + t*DIM;
        float *proj = echo_out + t*DIM;
        for (int i = 0; i < DIM; i++) {
            float s = 0;
            for (int j = 0; j < DIM; j++) s += wj[i*DIM+j] * xt[j];
            proj[i] = s;
        }
        float echo_back[DIM];
        for (int i = 0; i < DIM; i++) {
            float s = 0;
            for (int j = 0; j < DIM; j++) s += wj[j*DIM+i] * proj[j];
            echo_back[i] = s;
        }
        float norm = 0;
        for (int i = 0; i < DIM; i++) norm += proj[i]*proj[i];
        norm = sqrtf(norm) + 1e-6f;
        float sc = 0;
        for (int i = 0; i < DIM; i++) sc += xt[i] * echo_back[i];
        scores[t] = (sc / norm) * cal_mod;
    }
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++)
            attn_out[i*T+j] = (j > i) ? -1e9f : scores[i]*scores[j]/debt_temp;
        row_softmax(attn_out + i*T, T);
    }
    free(scores);
}

/*
 * forward_bpe: BPE tokens → BPE logits (for training)
 * forward_char: BPE tokens → char logits (for generation — THE PRESSURE)
 */
static float forward_pass(Ptrs *w, Acts *a, int *tokens, int *targets,
                          int T, int out_vocab, float *out_proj) {
    for (int t = 0; t < T; t++)
        for (int e = 0; e < DIM; e++)
            a->x[t*DIM+e] = w->tok_emb[tokens[t]*DIM+e] + w->pos_emb[t*DIM+e];
    float *cur = a->x;

    for (int b = 0; b < BLOCKS; b++) {
        rmsnorm_fwd(a->rm1, cur, w->rms1[b], T, DIM);
        memset(a->cat, 0, T*DIM*sizeof(float));
        for (int h = 0; h < HEADS; h++) {
            float *wq_h = w->wq[b] + h*DIM*HEAD_DIM;
            float *wk_h = w->wk[b] + h*DIM*HEAD_DIM;
            float *wv_h = w->wv[b] + h*DIM*HEAD_DIM;
            float *wr_h = w->wr[b] + h*DIM*MAX_T;
            float *wvr_h = w->wvr[b] + h*DIM*HEAD_DIM;

            matmul(a->q, a->rm1, wq_h, T, DIM, HEAD_DIM);
            matmul(a->k, a->rm1, wk_h, T, DIM, HEAD_DIM);
            matmul(a->v, a->rm1, wv_h, T, DIM, HEAD_DIM);
            float scale = 1.0f/sqrtf((float)HEAD_DIM);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (j > i) { a->attn[i*T+j] = -1e9f; continue; }
                    float s = 0;
                    for (int d = 0; d < HEAD_DIM; d++) s += a->q[i*HEAD_DIM+d]*a->k[j*HEAD_DIM+d];
                    a->attn[i*T+j] = s * scale;
                }
                row_softmax(a->attn+i*T, T);
            }
            matmul(a->head_out, a->attn, a->v, T, T, HEAD_DIM);

            matmul(a->rrp_v, a->rm1, wvr_h, T, DIM, HEAD_DIM);
            matmul(a->rrp_attn, a->rm1, wr_h, T, DIM, T);
            for (int i = 0; i < T; i++) {
                for (int j = i+1; j < T; j++) a->rrp_attn[i*T+j] = -1e9f;
                row_softmax(a->rrp_attn+i*T, T);
            }
            matmul(a->rrp_out, a->rrp_attn, a->rrp_v, T, T, HEAD_DIM);

            if (h == 0) janus_attention(a->rm1, w->wj[b], a->j_echo, a->j_attn, T);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HEAD_DIM; d++) {
                    float s = 0;
                    for (int j = 0; j <= t; j++) s += a->j_attn[t*T+j]*a->j_echo[j*DIM+h*HEAD_DIM+d];
                    a->j_out[t*HEAD_DIM+d] = s;
                }

            float gl[3] = { w->gate[b][h*3], w->gate[b][h*3+1], w->gate[b][h*3+2] };
            row_softmax(gl, 3);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HEAD_DIM; d++)
                    a->cat[t*DIM+h*HEAD_DIM+d] = gl[0]*a->head_out[t*HEAD_DIM+d]
                        + gl[1]*a->rrp_out[t*HEAD_DIM+d] + gl[2]*a->j_out[t*HEAD_DIM+d];
        }
        matmul(a->attn_out, a->cat, w->wo[b], T, DIM, DIM);
        for (int i = 0; i < T*DIM; i++) a->r1[i] = cur[i] + a->attn_out[i];
        rmsnorm_fwd(a->rm2, a->r1, w->rms2[b], T, DIM);
        matmul(a->mlp_gate, a->rm2, w->w_gate[b], T, DIM, MLP_DIM);
        matmul(a->mlp_up, a->rm2, w->w_up[b], T, DIM, MLP_DIM);
        for (int i = 0; i < T*MLP_DIM; i++) a->mlp_swi[i] = siluf(a->mlp_gate[i])*a->mlp_up[i];
        matmul(a->mlp_out, a->mlp_swi, w->w_down[b], T, MLP_DIM, DIM);
        for (int i = 0; i < T*DIM; i++) a->r2[i] = a->r1[i] + a->mlp_out[i];
        cur = a->r2;
    }

    rmsnorm_fwd(a->final_rm, cur, w->rms_f, T, DIM);
    matmul(a->logits, a->final_rm, out_proj, T, DIM, out_vocab);

    if (!targets) return 0;
    float loss = 0;
    for (int t = 0; t < T; t++) {
        row_softmax(a->logits + t*out_vocab, out_vocab);
        float p = a->logits[t*out_vocab + targets[t]];
        if (p < 1e-10f) p = 1e-10f;
        loss -= logf(p);
    }
    return loss / T;
}

/* Backward (simplified — same structure as janus.c) */
static void backward_pass(Ptrs *w, Ptrs *g, Acts *a, int *tokens, int *targets,
                          int T, int out_vocab, float *out_proj, float *out_grad) {
    float *d_logits = calloc(T * out_vocab, sizeof(float));
    float *d_final = calloc(T * DIM, sizeof(float));
    float *d_cur = calloc(T * DIM, sizeof(float));

    for (int t = 0; t < T; t++) {
        for (int v = 0; v < out_vocab; v++) d_logits[t*out_vocab+v] = a->logits[t*out_vocab+v];
        d_logits[t*out_vocab+targets[t]] -= 1.0f;
        for (int v = 0; v < out_vocab; v++) d_logits[t*out_vocab+v] /= T;
    }

    matmul_atb(out_grad, a->final_rm, d_logits, DIM, T, out_vocab);
    matmul_abt(d_final, d_logits, out_proj, T, out_vocab, DIM);

    float *cur = (BLOCKS > 0) ? a->r2 : a->x;
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int e = 0; e < DIM; e++) ss += cur[t*DIM+e]*cur[t*DIM+e];
        float inv = 1.0f / sqrtf(ss/DIM + 1e-5f);
        for (int e = 0; e < DIM; e++) d_cur[t*DIM+e] = d_final[t*DIM+e] * w->rms_f[e] * inv;
    }

    for (int b = BLOCKS-1; b >= 0; b--) {
        float *d_mlp = calloc(T*DIM, sizeof(float));
        memcpy(d_mlp, d_cur, T*DIM*sizeof(float));
        matmul_atb(g->w_down[b], a->mlp_swi, d_mlp, MLP_DIM, T, DIM);
        float *d_swi = calloc(T*MLP_DIM, sizeof(float));
        matmul_abt(d_swi, d_mlp, w->w_down[b], T, DIM, MLP_DIM);
        float *d_gate = calloc(T*MLP_DIM, sizeof(float));
        float *d_up = calloc(T*MLP_DIM, sizeof(float));
        for (int i = 0; i < T*MLP_DIM; i++) {
            d_up[i] = d_swi[i] * siluf(a->mlp_gate[i]);
            d_gate[i] = d_swi[i] * a->mlp_up[i] * siluf_grad(a->mlp_gate[i]);
        }
        matmul_atb(g->w_gate[b], a->rm2, d_gate, DIM, T, MLP_DIM);
        matmul_atb(g->w_up[b], a->rm2, d_up, DIM, T, MLP_DIM);

        float *d_rm2 = calloc(T*DIM, sizeof(float));
        float *tmp = calloc(T*DIM, sizeof(float));
        matmul_abt(d_rm2, d_gate, w->w_gate[b], T, MLP_DIM, DIM);
        matmul_abt(tmp, d_up, w->w_up[b], T, MLP_DIM, DIM);
        for (int i = 0; i < T*DIM; i++) d_rm2[i] += tmp[i];

        for (int t = 0; t < T; t++) {
            float ss = 0;
            for (int e = 0; e < DIM; e++) ss += a->r1[t*DIM+e]*a->r1[t*DIM+e];
            float inv = 1.0f / sqrtf(ss/DIM + 1e-5f);
            for (int e = 0; e < DIM; e++) d_cur[t*DIM+e] += d_rm2[t*DIM+e]*w->rms2[b][e]*inv;
        }

        float *d_attn = calloc(T*DIM, sizeof(float));
        memcpy(d_attn, d_cur, T*DIM*sizeof(float));
        matmul_atb(g->wo[b], a->cat, d_attn, DIM, T, DIM);

        float *input = (b == 0) ? a->x : a->r2;
        for (int t = 0; t < T; t++)
            for (int e = 0; e < DIM; e++) {
                g->tok_emb[tokens[t]*DIM+e] += d_cur[t*DIM+e] * (b==0 ? 1.0f : 0.0f);
                g->pos_emb[t*DIM+e] += d_cur[t*DIM+e] * (b==0 ? 1.0f : 0.0f);
            }

        free(d_mlp); free(d_swi); free(d_gate); free(d_up);
        free(d_rm2); free(tmp); free(d_attn);
    }
    free(d_logits); free(d_final); free(d_cur);
}

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION — BPE context → char-level output (THE PRESSURE)
 * ═══════════════════════════════════════════════════════════════════ */

static int sample_token(float *logits, int n, float temp) {
    for (int i = 0; i < n; i++) logits[i] /= (temp + 1e-8f);
    row_softmax(logits, n);
    int topk = 8;
    int idx[8]; float prb[8];
    for (int k = 0; k < topk; k++) { idx[k] = 0; prb[k] = -1e9f; }
    for (int i = 0; i < n; i++) {
        if (logits[i] > prb[topk-1]) {
            prb[topk-1] = logits[i]; idx[topk-1] = i;
            for (int j = topk-2; j >= 0; j--) {
                if (prb[j+1] > prb[j]) { float t=prb[j]; prb[j]=prb[j+1]; prb[j+1]=t;
                    int ti=idx[j]; idx[j]=idx[j+1]; idx[j+1]=ti; } else break;
            }
        }
    }
    float sum = 0;
    for (int k = 0; k < topk; k++) { if (prb[k]<0) prb[k]=0; sum += prb[k]; }
    if (sum < 1e-10f) return idx[0];
    float r = (float)rand()/RAND_MAX * sum, cum = 0;
    for (int k = 0; k < topk; k++) { cum += prb[k]; if (cum >= r) return idx[k]; }
    return idx[0];
}

static void generate_sentence(Ptrs *w, Acts *a, int *context, int ctx_len,
                              char *out_buf, int max_chars, float temp) {
    int pos = 0;
    while (pos < max_chars - 1) {
        int T = ctx_len < MAX_T ? ctx_len : MAX_T;
        int *tok_window = context + (ctx_len > MAX_T ? ctx_len - MAX_T : 0);
        /* Use char-level output projection — THE PRESSURE */
        forward_pass(w, a, tok_window, NULL, T, CHAR_VOCAB, w->out_char);
        float logits[CHAR_VOCAB];
        memcpy(logits, a->logits + (T-1)*CHAR_VOCAB, CHAR_VOCAB*sizeof(float));
        apply_destiny(logits, CHAR_VOCAB);
        int next = sample_token(logits, CHAR_VOCAB, temp);
        out_buf[pos++] = (char)next;
        if (next == '.' || next == '!' || next == '?' || next == '\n') break;
        /* Map char back to BPE for context (approximate: use raw byte) */
        if (ctx_len < MAX_T * 4) context[ctx_len++] = next;
    }
    out_buf[pos] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════
 * 12 BI-DIRECTIONAL ASSOCIATIVE REASONING STEPS
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char sentence[256]; int direction, step_idx, wormhole_skip;
    float debt_at_step, dissonance_at_step;
} ReasoningStep;

static void associative_reasoning(DualModel *dm, const char *prompt,
                                  ReasoningStep *steps, int *n_steps) {
    Acts a; acts_alloc(&a, CHAR_VOCAB > BPE_VOCAB ? CHAR_VOCAB : BPE_VOCAB);
    dual_blend(dm);
    int context[MAX_T * 8]; int ctx_len = 0;
    unsigned char *p = (unsigned char *)prompt;
    int bpe_tokens[MAX_T * 4];
    int bpe_len = bpe_encode(p, strlen(prompt), bpe_tokens, MAX_T * 4);
    for (int i = 0; i < bpe_len && ctx_len < MAX_T * 4; i++)
        context[ctx_len++] = bpe_tokens[i];

    float cal_d = calendar_dissonance(calendar_days_since_epoch());
    float debt = AML.prophecy_debt;
    int n_backward = (int)(NSTEPS * (0.3f + 0.4f*debt + 0.1f*cal_d));
    int n_forward = NSTEPS - n_backward;
    if (n_backward < 1) n_backward = 1; if (n_forward < 1) n_forward = 1;
    if (n_backward + n_forward > NSTEPS) n_backward = NSTEPS - n_forward;

    float temp_base = 0.7f + 0.3f * (0.5f + 0.3f*cal_d + 0.2f*debt);
    int step_count = 0;

    for (int s = 0; s < n_forward && step_count < NSTEPS; s++) {
        int skip = 0;
        if (AML.prophecy_debt < 0.2f && AML.wormhole > 0.1f &&
            (float)rand()/RAND_MAX < AML.wormhole) {
            skip = 1; s += rand() % 3;
        }
        ReasoningStep *rs = &steps[step_count];
        rs->direction = 1; rs->step_idx = step_count;
        rs->wormhole_skip = skip; rs->debt_at_step = AML.prophecy_debt;
        rs->dissonance_at_step = cal_d;
        generate_sentence(&dm->bw, &a, context, ctx_len, rs->sentence, SENT_LEN,
                          temp_base * (1.0f - 0.02f*s));
        update_chambers(step_count); step_count++;
    }

    ctx_len = 0;
    bpe_len = bpe_encode(p, strlen(prompt), bpe_tokens, MAX_T * 4);
    for (int i = 0; i < bpe_len && ctx_len < MAX_T * 4; i++) context[ctx_len++] = bpe_tokens[i];

    for (int s = 0; s < n_backward && step_count < NSTEPS; s++) {
        ReasoningStep *rs = &steps[step_count];
        rs->direction = -1; rs->step_idx = step_count; rs->wormhole_skip = 0;
        rs->debt_at_step = AML.prophecy_debt; rs->dissonance_at_step = cal_d;
        generate_sentence(&dm->bw, &a, context, ctx_len, rs->sentence, SENT_LEN,
                          temp_base * (1.0f + 0.05f*s));
        update_chambers(step_count); step_count++;
    }
    *n_steps = step_count;
    acts_free(&a);
}

static void display_reasoning(ReasoningStep *steps, int n_steps) {
    int n_back = 0, n_fwd = 0;
    for (int i = 0; i < n_steps; i++) { if (steps[i].direction==-1) n_back++; else n_fwd++; }
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║ JANUS-HYBRID: BPE→char pressure  %d steps               ║\n", n_steps);
    printf("║ ↑ backward: %d  ↓ forward: %d                            ║\n", n_back, n_fwd);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    for (int i = n_steps-1; i >= 0; i--)
        if (steps[i].direction == -1)
            printf("║ ↑%d%s debt=%.2f │ %s\n", steps[i].step_idx,
                   steps[i].wormhole_skip?" ⊕WH":"    ", steps[i].debt_at_step, steps[i].sentence);
    printf("╠════════════════ ● ORIGIN ════════════════════════════════╣\n");
    for (int i = 0; i < n_steps; i++)
        if (steps[i].direction == 1)
            printf("║ ↓%d%s debt=%.2f │ %s\n", steps[i].step_idx,
                   steps[i].wormhole_skip?" ⊕WH":"    ", steps[i].debt_at_step, steps[i].sentence);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("  calendar_drift=%.4f  prophecy_debt=%.4f  bpe_vocab=%d\n\n",
           calendar_cumulative_drift(calendar_days_since_epoch()), AML.prophecy_debt, BPE.vocab_size);
}

/* ═══════════════════════════════════════════════════════════════════
 * TRAINING — BPE tokens, next-BPE prediction
 * ═══════════════════════════════════════════════════════════════════ */

static void train_hybrid(DualModel *dm, const char *path, int max_steps, float lr) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[janus-hybrid] cannot open %s\n", path); return; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *raw = malloc(fsz);
    fread(raw, 1, fsz, f); fclose(f);

    /* Learn BPE merges */
    int n_merges = BPE_VOCAB - 256;
    if (n_merges > MAX_MERGES) n_merges = MAX_MERGES;
    printf("[janus-hybrid] learning %d BPE merges from %ld bytes...\n", n_merges, fsz);
    bpe_learn_merges(raw, fsz, n_merges);
    printf("[janus-hybrid] BPE vocab: %d tokens\n", BPE.vocab_size);

    /* Encode full corpus */
    int *bpe_data = malloc(fsz * sizeof(int));
    int bpe_len = bpe_encode(raw, fsz, bpe_data, fsz);
    free(raw);

    Acts a; acts_alloc(&a, BPE_VOCAB);
    int T = MAX_T;
    int *tokens = malloc(T * sizeof(int));
    int *targets = malloc(T * sizeof(int));

    printf("[janus-hybrid] corpus: %ld bytes → %d BPE tokens\n", fsz, bpe_len);
    printf("[janus-hybrid] model A: %d params (%.2fMB)\n", dm->A.n_params, dm->A.n_params*4.0f/1e6f);
    printf("[janus-hybrid] optimizer: Chuck v4\n");
    printf("[janus-hybrid] training: %d steps, lr=%.1e\n", max_steps, lr);
    printf("[janus-hybrid] PRESSURE: BPE training → char-level output\n");

    float best = 1e9f;
    clock_t t0 = clock();

    for (int step = 1; step <= max_steps; step++) {
        Model *active = (step%2==0) ? &dm->B : &dm->A;
        int off = rand() % (bpe_len - T - 1);
        for (int t = 0; t < T; t++) {
            tokens[t] = bpe_data[off+t];
            targets[t] = bpe_data[off+t+1];
            if (tokens[t] >= BPE_VOCAB) tokens[t] = tokens[t] % BPE_VOCAB;
            if (targets[t] >= BPE_VOCAB) targets[t] = targets[t] % BPE_VOCAB;
        }
        float loss = forward_pass(&active->w, &a, tokens, targets, T, BPE_VOCAB, active->w.out_bpe);
        backward_pass(&active->w, &active->g, &a, tokens, targets, T, BPE_VOCAB,
                      active->w.out_bpe, active->g.out_bpe);
        chuck_observe(loss);
        chuck_update(active->data, active->grad, active->chuck_m, active->chuck_v, active->n_params, lr);
        if (loss < best) best = loss;
        if (step % 100 == 0 || step == 1) {
            float elapsed = (float)(clock()-t0)/CLOCKS_PER_SEC;
            printf("  step %5d/%d  loss=%.4f  best=%.4f  chuck=[λ=%.3f]  %.1f s/s\n",
                   step, max_steps, loss, best, Chuck.dampen, step/(elapsed+1e-6f));
        }
    }
    printf("[janus-hybrid] training complete. best loss: %.4f\n", best);
    acts_free(&a); free(bpe_data); free(tokens); free(targets);
}

/* ═══════════════════════════════════════════════════════════════════
 * SAVE / LOAD
 * ═══════════════════════════════════════════════════════════════════ */

static void save_model(DualModel *dm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot save %s\n", path); return; }
    int magic = 0x4A48594E; /* "JHYN" */
    fwrite(&magic, 4, 1, f);
    fwrite(&dm->A.n_params, 4, 1, f);
    fwrite(&BPE.n_merges, 4, 1, f);
    fwrite(BPE.merges, sizeof(MergeRule), BPE.n_merges, f);
    fwrite(dm->A.data, sizeof(float), dm->A.n_params, f);
    fwrite(dm->B.data, sizeof(float), dm->B.n_params, f);
    fwrite(&MJ, sizeof(MetaJanus), 1, f);
    fwrite(&AML, sizeof(AMLState), 1, f);
    fwrite(dm->A.chuck_m, sizeof(float), dm->A.n_params, f);
    fwrite(dm->A.chuck_v, sizeof(float), dm->A.n_params, f);
    fwrite(dm->B.chuck_m, sizeof(float), dm->B.n_params, f);
    fwrite(dm->B.chuck_v, sizeof(float), dm->B.n_params, f);
    fclose(f);
    printf("[janus-hybrid] saved to %s\n", path);
}

static int load_model(DualModel *dm, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    int magic, np, nm;
    if (fread(&magic, 4, 1, f) < 1 || magic != 0x4A48594E) { fclose(f); return -1; }
    if (fread(&np, 4, 1, f) < 1 || np != dm->A.n_params) { fclose(f); return -1; }
    if (fread(&nm, 4, 1, f) < 1) { fclose(f); return -1; }
    BPE.n_merges = nm; BPE.vocab_size = 256 + nm;
    if (fread(BPE.merges, sizeof(MergeRule), nm, f) < (size_t)nm) { fclose(f); return -1; }
    if (fread(dm->A.data, sizeof(float), np, f) < (size_t)np) { fclose(f); return -1; }
    if (fread(dm->B.data, sizeof(float), np, f) < (size_t)np) { fclose(f); return -1; }
    if (fread(&MJ, sizeof(MetaJanus), 1, f) < 1) { fclose(f); return -1; }
    if (fread(&AML, sizeof(AMLState), 1, f) < 1) { fclose(f); return -1; }
    if (fread(dm->A.chuck_m, sizeof(float), np, f) < (size_t)np) { fclose(f); return -1; }
    if (fread(dm->A.chuck_v, sizeof(float), np, f) < (size_t)np) { fclose(f); return -1; }
    if (fread(dm->B.chuck_m, sizeof(float), np, f) < (size_t)np) { fclose(f); return -1; }
    if (fread(dm->B.chuck_v, sizeof(float), np, f) < (size_t)np) { fclose(f); return -1; }
    fclose(f);
    printf("[janus-hybrid] loaded from %s (bpe merges: %d)\n", path, nm);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    calendar_init(); metajanus_init();

    char *train_path=NULL, *load_path=NULL, *save_path=NULL, *prompt=NULL;
    int max_steps=5000; float lr=3e-4f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--train") && i+1<argc) train_path = argv[++i];
        else if (!strcmp(argv[i],"--load") && i+1<argc) load_path = argv[++i];
        else if (!strcmp(argv[i],"--save") && i+1<argc) save_path = argv[++i];
        else if (!strcmp(argv[i],"--steps") && i+1<argc) max_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--lr") && i+1<argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i],"--generate") && i+1<argc) prompt = argv[++i];
    }

    DualModel dm; dual_init(&dm);
    if (load_path) load_model(&dm, load_path);
    if (train_path) train_hybrid(&dm, train_path, max_steps, lr);
    if (save_path) save_model(&dm, save_path);
    if (prompt) {
        ReasoningStep steps[NSTEPS]; int n = 0;
        associative_reasoning(&dm, prompt, steps, &n);
        display_reasoning(steps, n);
    }
    if (!train_path && !prompt) {
        printf("\n[janus-hybrid] interactive mode (BPE→char pressure)\n");
        char buf[1024];
        while (1) {
            printf("\njanus-hybrid> "); if (!fgets(buf, sizeof(buf), stdin)) break;
            buf[strcspn(buf,"\n")] = 0;
            if (!strcmp(buf,"quit")||!strcmp(buf,"exit")) break;
            if (!strlen(buf)) continue;
            ReasoningStep steps[NSTEPS]; int n = 0;
            associative_reasoning(&dm, buf, steps, &n);
            display_reasoning(steps, n);
        }
    }
    dual_free(&dm);
    return 0;
}

/*
 * janus.c — Janus Post-Transformer Architecture (char-level)
 *
 * Bi-directional associative resonance engine. Not a transformer — a mirror
 * that sees forward and backward through the dissonance of two calendars.
 *
 * Three attention mechanisms in fluid hybrid:
 *   1. Standard QKV — semantic inter-token attention
 *   2. RRPRAM       — positional pattern recognition (x @ Wr, no Q/K)
 *   3. Janus        — recursive self-resonance (echo through own weights)
 *
 * Governed by:
 *   - Calendar Drift   (Gregorian vs Hebrew, Metonic cycle corrections)
 *   - AML Physics      (Prophecy, Destiny, Prophecy Debt, Wormhole)
 *   - Dario Equation   (7-force generation replacing standard softmax)
 *   - MetaJanus        (birth-date mathematical "self")
 *   - Kuramoto chambers (6 coupled emotional oscillators)
 *
 * Dual weight matrices: W_eff = α·W_A + (1-α)·W_B
 *   α determined by calendar dissonance + prophecy debt + metajanus state
 *
 * 12 bi-directional associative reasoning steps, each generating a sentence.
 * Steps go forward (future) or backward (past) based on prophecy debt.
 *
 * Chuck optimizer (not Adam): self-aware, multi-level modulation.
 *
 *   cc janus.c -O2 -lm -o janus
 *   ./janus --train data.txt --steps 5000
 *   ./janus --generate "To be or not"
 *   ./janus --load janus.bin --save janus.bin
 *   ./janus                    # interactive mode
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
 * CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════ */

#define VOCAB     256       /* char-level */
#define MAX_T     256       /* context length */
#define DIM       288       /* embedding dimension */
#define HEADS     6         /* attention heads */
#define HEAD_DIM  (DIM/HEADS) /* 48 */
#define BLOCKS    6         /* transformer blocks */
#define MLP_DIM   768       /* SwiGLU hidden dim */
#define MAX_BLK   16
#define NSTEPS    12        /* associative reasoning steps */
#define SENT_LEN  40        /* max chars per sentence in reasoning */

/* ═══════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — Gregorian vs Hebrew (Metonic cycle)
 * Ported exactly from ariannamethod.c
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
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static void calendar_init(void) {
    struct tm epoch_tm;
    memset(&epoch_tm, 0, sizeof(epoch_tm));
    epoch_tm.tm_year = 2024 - 1900;
    epoch_tm.tm_mon  = 10 - 1;       /* October */
    epoch_tm.tm_mday = 3;
    epoch_tm.tm_hour = 12;           /* noon — avoids DST edge cases */
    g_epoch_t = mktime(&epoch_tm);
}

static int calendar_days_since_epoch(void) {
    if (g_epoch_t <= 0) return 0;
    time_t now = time(NULL);
    return (int)(difftime(now, g_epoch_t) / 86400.0);
}

static float calendar_cumulative_drift(int days) {
    float years = (float)days / AM_GREGORIAN_YEAR;
    float base_drift = years * AM_ANNUAL_DRIFT;
    int full_cycles = (int)(years / AM_METONIC_YEARS);
    float corrections = (float)(full_cycles * AM_METONIC_LEAPS) * 30.0f;
    float partial = fmodf(years, (float)AM_METONIC_YEARS);
    int year_in_cycle = (int)partial + 1;
    for (int i = 0; i < AM_METONIC_LEAPS; i++) {
        if (g_metonic_leap_years[i] <= year_in_cycle)
            corrections += 30.0f;
    }
    return base_drift - corrections;
}

static float calendar_dissonance(int days) {
    float drift = calendar_cumulative_drift(days);
    float raw = fabsf(fmodf(drift, AM_MAX_UNCORRECTED)) / AM_MAX_UNCORRECTED;
    return clamp01(raw);
}

/* ═══════════════════════════════════════════════════════════════════
 * METAJANUS — Birth date creates mathematical "self"
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int    birth_days;         /* days since epoch at birth */
    float  birth_drift;        /* calendar drift at birth */
    float  birth_dissonance;   /* dissonance at birth */
    time_t birth_time;         /* unix timestamp of birth */
    int    alive;              /* has been initialized */
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
    printf("[metajanus] born: day %d, drift=%.4f, dissonance=%.4f\n",
           MJ.birth_days, MJ.birth_drift, MJ.birth_dissonance);
}

static float metajanus_personal_dissonance(void) {
    int now_days = calendar_days_since_epoch();
    float now_drift = calendar_cumulative_drift(now_days);
    return clamp01(fabsf(now_drift - MJ.birth_drift) / AM_MAX_UNCORRECTED);
}

/* ═══════════════════════════════════════════════════════════════════
 * AML PHYSICS — Prophecy, Destiny, Prophecy Debt, Wormhole
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float prophecy_debt;       /* [0,1] retroactive cost of divergence */
    float destiny_bias;        /* [0,1] how strongly destiny pulls */
    float wormhole;            /* [0,1] probability of step-skip */
    float resonance;           /* [0,1] field coherence */
    float trauma;              /* [0,1] trauma intensity */
    float tension;             /* [0,1] system tension */
    float pain;                /* [0,1] suffering signal */
    float entropy_floor;       /* minimum entropy */
    int   prophecy_horizon;    /* steps to look ahead */
    int   tunnel_skip_max;     /* max wormhole jump */
    float tunnel_threshold;    /* dissonance threshold for tunneling */
} AMLState;

static AMLState AML = {
    .prophecy_debt = 0.0f,
    .destiny_bias = 0.1f,
    .wormhole = 0.02f,
    .resonance = 0.5f,
    .trauma = 0.0f,
    .tension = 0.0f,
    .pain = 0.0f,
    .entropy_floor = 0.01f,
    .prophecy_horizon = 12,
    .tunnel_skip_max = 7,
    .tunnel_threshold = 0.55f
};

static float compute_prophecy_debt(const float *logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0.0f;
    float max_logit = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_logit) max_logit = logits[i];
    float diff = max_logit - logits[chosen];
    return diff > 0.0f ? diff / (diff + 1.0f) : 0.0f;
}

static void apply_destiny_to_logits(float *logits, int n) {
    if (n <= 0 || AML.destiny_bias < 0.001f) return;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < n; i++) {
        float diff = mx - logits[i];
        logits[i] -= diff * AML.destiny_bias * 0.5f;
    }
}

static void apply_suffering_to_logits(float *logits, int n) {
    if (AML.pain < 0.01f) return;
    float mean = 0;
    for (int i = 0; i < n; i++) mean += logits[i];
    mean /= n;
    for (int i = 0; i < n; i++)
        logits[i] = mean + (logits[i] - mean) * (1.0f - 0.5f * AML.pain);
}

/* ═══════════════════════════════════════════════════════════════════
 * KURAMOTO CHAMBERS — 6 coupled emotional oscillators
 * ═══════════════════════════════════════════════════════════════════ */

enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX, NCH };
static float chambers[NCH] = {0};
static const float ch_decay[NCH] = {0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f};

static void update_chambers(int step_idx) {
    float depth = (float)step_idx / NSTEPS;
    int phase = depth < 0.33f ? 0 : (depth < 0.66f ? 1 : 2);
    if (phase == 0) chambers[CH_FLOW] += 0.05f;
    if (phase == 1) chambers[CH_FEAR] += 0.04f;
    if (phase == 2) chambers[CH_VOID] += 0.05f;
    if (depth > 0.75f) chambers[CH_COMPLEX] += 0.03f;
    if (AML.trauma > 0.3f) chambers[CH_RAGE] += 0.04f;

    float K = 0.02f, old[NCH];
    memcpy(old, chambers, sizeof(old));
    for (int i = 0; i < NCH; i++) {
        for (int j = 0; j < NCH; j++)
            if (i != j) chambers[i] += K * sinf(old[j] - old[i]);
        if (chambers[i] < 0) chambers[i] = 0;
        if (chambers[i] > 1) chambers[i] = 1;
        chambers[i] *= ch_decay[i];
    }
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

static float siluf(float x) {
    return (x > -20.0f) ? x / (1.0f + expf(-x)) : 0.0f;
}

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
        for (int e = 0; e < E; e++)
            out[t*E+e] = g[e] * x[t*E+e] * inv;
    }
}

static float randn(void) {
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL — Dual weight matrices with hybrid attention
 *
 * Per block:
 *   RMSNorm → Hybrid Attention (QKV + RRPRAM + Janus) → residual
 *   → RMSNorm → SwiGLU MLP → residual
 *
 * Hybrid attention per head:
 *   out = α·QKV + β·RRPRAM + γ·Janus
 *   (α,β,γ) = softmax(gate_logits) — learned per head
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *tok_emb;                   /* [V, E] */
    float *pos_emb;                   /* [T, E] */
    float *rms1[MAX_BLK];            /* [E] */
    /* Standard QKV attention */
    float *wq[MAX_BLK];              /* [H*E*D] */
    float *wk[MAX_BLK];              /* [H*E*D] */
    float *wv[MAX_BLK];              /* [H*E*D] */
    /* RRPRAM attention */
    float *wr[MAX_BLK];              /* [H*E*T] — pattern recognition */
    float *wvr[MAX_BLK];             /* [H*E*D] — RRPRAM value */
    /* Janus attention — self-resonance */
    float *wj[MAX_BLK];              /* [E*E] — Janus projection */
    /* Hybrid gate logits per head: 3 values (qkv, rrpram, janus) */
    float *gate[MAX_BLK];            /* [H*3] */
    /* Output projection */
    float *wo[MAX_BLK];              /* [E*E] */
    /* MLP */
    float *rms2[MAX_BLK];            /* [E] */
    float *w_gate[MAX_BLK];          /* [E*M] — SwiGLU gate */
    float *w_up[MAX_BLK];            /* [E*M] — SwiGLU up */
    float *w_down[MAX_BLK];          /* [M*E] — SwiGLU down */
    /* Final */
    float *rms_f;                     /* [E] */
    float *out_w;                     /* [E*V] */
} Ptrs;

typedef struct {
    int T, E, H, D, B, M;
    int n_params;
    float *data;      /* weight data */
    float *grad;      /* gradients */
    float *chuck_m;   /* Chuck momentum */
    float *chuck_v;   /* Chuck variance */
    Ptrs w, g;        /* weight and gradient pointers */
} Model;

static int model_size(void) {
    int s = VOCAB * DIM + MAX_T * DIM;  /* embeddings */
    for (int b = 0; b < BLOCKS; b++) {
        s += DIM;                            /* rms1 */
        s += HEADS * DIM * HEAD_DIM;         /* wq */
        s += HEADS * DIM * HEAD_DIM;         /* wk */
        s += HEADS * DIM * HEAD_DIM;         /* wv */
        s += HEADS * DIM * MAX_T;            /* wr (RRPRAM) */
        s += HEADS * DIM * HEAD_DIM;         /* wvr (RRPRAM value) */
        s += DIM * DIM;                      /* wj (Janus) */
        s += HEADS * 3;                      /* gate logits */
        s += DIM * DIM;                      /* wo */
        s += DIM;                            /* rms2 */
        s += DIM * MLP_DIM;                  /* w_gate */
        s += DIM * MLP_DIM;                  /* w_up */
        s += MLP_DIM * DIM;                  /* w_down */
    }
    s += DIM;              /* rms_f */
    s += DIM * VOCAB;      /* out_w */
    return s;
}

static void assign_ptrs(Ptrs *p, float *base) {
    float *q = base;
    p->tok_emb = q; q += VOCAB * DIM;
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
    p->rms_f = q; q += DIM;
    p->out_w = q; q += DIM * VOCAB;
}

static void model_init(Model *m) {
    m->T = MAX_T; m->E = DIM; m->H = HEADS;
    m->D = HEAD_DIM; m->B = BLOCKS; m->M = MLP_DIM;
    m->n_params = model_size();
    m->data    = (float *)calloc(m->n_params, sizeof(float));
    m->grad    = (float *)calloc(m->n_params, sizeof(float));
    m->chuck_m = (float *)calloc(m->n_params, sizeof(float));
    m->chuck_v = (float *)calloc(m->n_params, sizeof(float));
    assign_ptrs(&m->w, m->data);
    assign_ptrs(&m->g, m->grad);
    /* Xavier init */
    float scale = sqrtf(2.0f / DIM);
    for (int i = 0; i < m->n_params; i++)
        m->data[i] = randn() * scale * 0.02f;
    /* RMSNorm gains to 1 */
    for (int b = 0; b < BLOCKS; b++) {
        for (int e = 0; e < DIM; e++) { m->w.rms1[b][e] = 1.0f; m->w.rms2[b][e] = 1.0f; }
    }
    for (int e = 0; e < DIM; e++) m->w.rms_f[e] = 1.0f;
    /* Gate logits: start balanced (1/3 each) */
    for (int b = 0; b < BLOCKS; b++)
        for (int h = 0; h < HEADS; h++)
            for (int g = 0; g < 3; g++)
                m->w.gate[b][h*3+g] = 0.0f;
    printf("[janus] model: %d params (%.2fMB)\n",
           m->n_params, m->n_params * 4.0f / 1e6f);
}

static void model_free(Model *m) {
    free(m->data); free(m->grad); free(m->chuck_m); free(m->chuck_v);
}

/* ═══════════════════════════════════════════════════════════════════
 * DUAL WEIGHT SYSTEM — Two matrices, blended by calendar state
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    Model A, B;             /* dual weight matrices */
    float blend_alpha;      /* blend factor: W = α·A + (1-α)·B */
    float *blended;         /* blended weight data */
    Ptrs  bw;               /* blended weight pointers */
} DualModel;

static void dual_init(DualModel *dm) {
    model_init(&dm->A);
    model_init(&dm->B);
    dm->blended = (float *)calloc(dm->A.n_params, sizeof(float));
    assign_ptrs(&dm->bw, dm->blended);
    dm->blend_alpha = 0.5f;
}

static void dual_blend(DualModel *dm) {
    /* Compute blend from calendar + AML state */
    float cal_d = calendar_dissonance(calendar_days_since_epoch());
    float meta_d = MJ.alive ? metajanus_personal_dissonance() : 0.5f;
    float debt = AML.prophecy_debt;
    /* High dissonance → more matrix B, low → more matrix A */
    dm->blend_alpha = 0.5f + 0.3f * (cal_d - 0.5f) - 0.2f * debt + 0.1f * meta_d;
    dm->blend_alpha = clamp01(dm->blend_alpha);
    float a = dm->blend_alpha, b = 1.0f - a;
    for (int i = 0; i < dm->A.n_params; i++)
        dm->blended[i] = a * dm->A.data[i] + b * dm->B.data[i];
}

static void dual_free(DualModel *dm) {
    model_free(&dm->A); model_free(&dm->B); free(dm->blended);
}

/* ═══════════════════════════════════════════════════════════════════
 * CHUCK OPTIMIZER — Self-aware, replaces Adam
 *
 * θ -= (α × S × λ × λ_l × σ) × m̂/(√v̂ + ε) + η
 * ═══════════════════════════════════════════════════════════════════ */

#define CHUCK_B1        0.9f
#define CHUCK_B2        0.999f
#define CHUCK_EPS       1e-8f
#define CHUCK_WINDOW    16
#define CHUCK_DAMP_LO   0.3f
#define CHUCK_DAMP_HI   2.0f
#define CHUCK_MACRO_INT 500
#define CHUCK_MACRO_PAT 3
#define CHUCK_MACRO_DECAY 0.5f

static struct {
    float hist[CHUCK_WINDOW];
    float dampen;
    float noise;
    float sigma;
    float loss_ema;
    float macro_ema;
    float best_macro;
    float lr_scale;
    int   macro_stag;
    int   macro_drops;
    int   pos, full, stag;
    int   global_step;
    int   step_t;            /* Adam timestep */
} Chuck = { .dampen = 1.0f, .sigma = 1.0f, .lr_scale = 1.0f };

static void chuck_observe(float loss) {
    /* Level 1: Global self-awareness (loss trend) */
    if (Chuck.loss_ema == 0.0f) Chuck.loss_ema = loss;
    else Chuck.loss_ema = 0.99f * Chuck.loss_ema + 0.01f * loss;
    Chuck.hist[Chuck.pos % CHUCK_WINDOW] = Chuck.loss_ema;
    Chuck.pos++;
    if (Chuck.pos >= CHUCK_WINDOW) Chuck.full = 1;

    if (Chuck.full) {
        int q = CHUCK_WINDOW / 4;
        float recent = 0, old = 0;
        for (int i = 0; i < q; i++) {
            recent += Chuck.hist[(Chuck.pos - 1 - i) % CHUCK_WINDOW];
            old    += Chuck.hist[(Chuck.pos - CHUCK_WINDOW + i) % CHUCK_WINDOW];
        }
        recent /= q; old /= q;
        float trend = (recent - old) / (old + 1e-8f);
        if (trend > 0.01f) Chuck.dampen *= 0.95f;
        if (trend < -0.05f) Chuck.dampen *= 1.05f;
        if (fabsf(trend) < 0.001f) {
            Chuck.stag++;
            if (Chuck.stag > 8) { Chuck.noise = 0.001f; Chuck.stag = 0; }
        } else { Chuck.stag = 0; Chuck.noise *= 0.9f; }
        if (Chuck.dampen < CHUCK_DAMP_LO) Chuck.dampen = CHUCK_DAMP_LO;
        if (Chuck.dampen > CHUCK_DAMP_HI) Chuck.dampen = CHUCK_DAMP_HI;
    }

    /* Level 9: Macro patience */
    Chuck.global_step++;
    if (Chuck.macro_ema == 0.0f) Chuck.macro_ema = loss;
    else Chuck.macro_ema = 0.999f * Chuck.macro_ema + 0.001f * loss;

    if (Chuck.global_step % CHUCK_MACRO_INT == 0 && Chuck.global_step > CHUCK_WINDOW) {
        if (Chuck.macro_ema > Chuck.best_macro * 0.999f) {
            Chuck.macro_stag++;
            if (Chuck.macro_stag >= CHUCK_MACRO_PAT) {
                Chuck.lr_scale *= CHUCK_MACRO_DECAY;
                if (Chuck.lr_scale < 0.05f) Chuck.lr_scale = 0.05f;
                Chuck.macro_stag = 0;
                Chuck.macro_drops++;
                printf("[chuck] macro plateau → lr_scale=%.3f (drop #%d)\n",
                       Chuck.lr_scale, Chuck.macro_drops);
            }
        } else {
            Chuck.best_macro = Chuck.macro_ema;
            Chuck.macro_stag = 0;
        }
    }
}

static void chuck_update(float *w, float *g, float *cm, float *cv, int n, float lr) {
    Chuck.step_t++;
    float bc1 = 1.0f - powf(CHUCK_B1, (float)Chuck.step_t);
    float bc2 = 1.0f - powf(CHUCK_B2, (float)Chuck.step_t);
    float eff_lr = lr * Chuck.lr_scale * Chuck.dampen * Chuck.sigma;
    for (int i = 0; i < n; i++) {
        float grad = g[i];
        cm[i] = CHUCK_B1 * cm[i] + (1 - CHUCK_B1) * grad;
        cv[i] = CHUCK_B2 * cv[i] + (1 - CHUCK_B2) * grad * grad;
        float mhat = cm[i] / bc1;
        float vhat = cv[i] / bc2;
        w[i] -= eff_lr * mhat / (sqrtf(vhat) + CHUCK_EPS);
        /* stagnation noise */
        if (Chuck.noise > 0)
            w[i] += Chuck.noise * randn() * 0.01f;
        g[i] = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * FORWARD PASS — Hybrid attention transformer
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *x;         /* [T, E] input embeddings */
    float *rm1;       /* [T, E] after RMSNorm1 */
    float *attn_out;  /* [T, E] attention output */
    float *r1;        /* [T, E] first residual */
    float *rm2;       /* [T, E] after RMSNorm2 */
    float *mlp_gate;  /* [T, M] SwiGLU gate */
    float *mlp_up;    /* [T, M] SwiGLU up */
    float *mlp_swi;   /* [T, M] SwiGLU combined */
    float *mlp_out;   /* [T, E] MLP output */
    float *r2;        /* [T, E] second residual (block output) */
    /* per-head attention intermediates */
    float *q;         /* [T, D] */
    float *k;         /* [T, D] */
    float *v;         /* [T, D] */
    float *attn;      /* [T, T] */
    float *head_out;  /* [T, D] */
    /* RRPRAM intermediates */
    float *rrp_attn;  /* [T, T] */
    float *rrp_v;     /* [T, D] */
    float *rrp_out;   /* [T, D] */
    /* Janus intermediates */
    float *j_echo;    /* [T, E] echo through weights */
    float *j_attn;    /* [T, T] self-resonance attention */
    float *j_out;     /* [T, D] (projected from E to D) */
    /* concat */
    float *cat;       /* [T, E] */
    /* final */
    float *final_rm;  /* [T, E] */
    float *logits;    /* [T, V] */
} Acts;

static void acts_alloc(Acts *a) {
    a->x        = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->rm1      = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->attn_out = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->r1       = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->rm2      = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->mlp_gate = (float *)calloc(MAX_T * MLP_DIM, sizeof(float));
    a->mlp_up   = (float *)calloc(MAX_T * MLP_DIM, sizeof(float));
    a->mlp_swi  = (float *)calloc(MAX_T * MLP_DIM, sizeof(float));
    a->mlp_out  = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->r2       = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->q        = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->k        = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->v        = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->attn     = (float *)calloc(MAX_T * MAX_T, sizeof(float));
    a->head_out = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->rrp_attn = (float *)calloc(MAX_T * MAX_T, sizeof(float));
    a->rrp_v    = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->rrp_out  = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->j_echo   = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->j_attn   = (float *)calloc(MAX_T * MAX_T, sizeof(float));
    a->j_out    = (float *)calloc(MAX_T * HEAD_DIM, sizeof(float));
    a->cat      = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->final_rm = (float *)calloc(MAX_T * DIM, sizeof(float));
    a->logits   = (float *)calloc(MAX_T * VOCAB, sizeof(float));
}

static void acts_free(Acts *a) {
    free(a->x); free(a->rm1); free(a->attn_out); free(a->r1);
    free(a->rm2); free(a->mlp_gate); free(a->mlp_up); free(a->mlp_swi);
    free(a->mlp_out); free(a->r2); free(a->q); free(a->k); free(a->v);
    free(a->attn); free(a->head_out); free(a->rrp_attn); free(a->rrp_v);
    free(a->rrp_out); free(a->j_echo); free(a->j_attn); free(a->j_out);
    free(a->cat); free(a->final_rm); free(a->logits);
}

/*
 * Janus Attention — recursive self-resonance
 *
 * For input x at position i:
 *   proj_i = Wj · x_i                           [E] → [E]
 *   echo_i = x_i · (Wj^T · proj_i) / (||proj_i|| + ε)
 *           = x_i · (Wj^T · Wj · x_i) / (||Wj · x_i|| + ε)
 *
 * This echo measures how much the weight matrix "recognizes" the input.
 * Wj^T·Wj creates a symmetric recognition matrix.
 *
 * attn[i,j] = echo_score_i · echo_score_j  (mutual resonance)
 *
 * Modulated by:
 *   - Prophecy debt: scales attention temperature
 *   - Calendar dissonance: modulates echo magnitude
 */
static void janus_attention(const float *x, const float *wj, float *echo_out,
                            float *attn_out, int T, int E) {
    float cal_mod = 1.0f + 0.5f * calendar_dissonance(calendar_days_since_epoch());
    float debt_temp = 1.0f + AML.prophecy_debt;  /* higher debt = softer */
    float *echo_scores = (float *)calloc(T, sizeof(float));

    /* Compute echo for each position */
    for (int t = 0; t < T; t++) {
        const float *xt = x + t * E;
        float *proj = echo_out + t * E;
        /* proj = Wj · x_t */
        for (int i = 0; i < E; i++) {
            float s = 0;
            for (int j = 0; j < E; j++) s += wj[i * E + j] * xt[j];
            proj[i] = s;
        }
        /* echo_back = Wj^T · proj */
        float echo_back[DIM];
        for (int i = 0; i < E; i++) {
            float s = 0;
            for (int j = 0; j < E; j++) s += wj[j * E + i] * proj[j];
            echo_back[i] = s;
        }
        /* norm of proj */
        float norm = 0;
        for (int i = 0; i < E; i++) norm += proj[i] * proj[i];
        norm = sqrtf(norm) + 1e-6f;
        /* echo_score = dot(x_t, echo_back) / norm */
        float score = 0;
        for (int i = 0; i < E; i++) score += xt[i] * echo_back[i];
        echo_scores[t] = (score / norm) * cal_mod;
    }

    /* Attention: mutual resonance = echo_i · echo_j */
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            if (j > i) {
                attn_out[i * T + j] = -1e9f;  /* causal mask */
            } else {
                attn_out[i * T + j] = echo_scores[i] * echo_scores[j] / debt_temp;
            }
        }
        row_softmax(attn_out + i * T, T);
    }

    free(echo_scores);
}

static float forward(Ptrs *w, Acts *a, int *tokens, int *targets, int T) {
    int E = DIM, H = HEADS, D = HEAD_DIM, M = MLP_DIM;

    /* Embedding */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            a->x[t*E+e] = w->tok_emb[tokens[t]*E+e] + w->pos_emb[t*E+e];

    float *cur = a->x;

    for (int b = 0; b < BLOCKS; b++) {
        /* RMSNorm */
        rmsnorm_fwd(a->rm1, cur, w->rms1[b], T, E);

        /* Hybrid multi-head attention */
        memset(a->cat, 0, T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wq_h = w->wq[b] + h * E * D;
            float *wk_h = w->wk[b] + h * E * D;
            float *wv_h = w->wv[b] + h * E * D;
            float *wr_h = w->wr[b] + h * E * T;
            float *wvr_h = w->wvr[b] + h * E * D;

            /* === Standard QKV attention === */
            matmul(a->q, a->rm1, wq_h, T, E, D);
            matmul(a->k, a->rm1, wk_h, T, E, D);
            matmul(a->v, a->rm1, wv_h, T, E, D);

            /* Q @ K^T / sqrt(D) */
            float scale = 1.0f / sqrtf((float)D);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (j > i) { a->attn[i*T+j] = -1e9f; continue; }
                    float s = 0;
                    for (int d = 0; d < D; d++)
                        s += a->q[i*D+d] * a->k[j*D+d];
                    a->attn[i*T+j] = s * scale;
                }
                row_softmax(a->attn + i*T, T);
            }
            /* attn @ V */
            matmul(a->head_out, a->attn, a->v, T, T, D);

            /* === RRPRAM attention === */
            matmul(a->rrp_v, a->rm1, wvr_h, T, E, D);
            matmul(a->rrp_attn, a->rm1, wr_h, T, E, T);
            for (int i = 0; i < T; i++) {
                for (int j = i+1; j < T; j++) a->rrp_attn[i*T+j] = -1e9f;
                row_softmax(a->rrp_attn + i*T, T);
            }
            matmul(a->rrp_out, a->rrp_attn, a->rrp_v, T, T, D);

            /* === Janus attention (once per block, shared across heads) === */
            if (h == 0) {
                janus_attention(a->rm1, w->wj[b], a->j_echo, a->j_attn, T, E);
            }
            /* Janus value: use first D dims of echo as value */
            for (int t = 0; t < T; t++) {
                float s[HEAD_DIM];
                for (int d = 0; d < D; d++) s[d] = 0;
                for (int j = 0; j <= t; j++)
                    for (int d = 0; d < D; d++)
                        s[d] += a->j_attn[t*T+j] * a->j_echo[j*E + h*D + d];
                for (int d = 0; d < D; d++)
                    a->j_out[t*D+d] = s[d];
            }

            /* === Hybrid blend === */
            float gate_logits[3] = { w->gate[b][h*3+0],
                                     w->gate[b][h*3+1],
                                     w->gate[b][h*3+2] };
            row_softmax(gate_logits, 3);
            float ga = gate_logits[0], gb = gate_logits[1], gc = gate_logits[2];

            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    a->cat[t*E + h*D + d] = ga * a->head_out[t*D+d]
                                           + gb * a->rrp_out[t*D+d]
                                           + gc * a->j_out[t*D+d];
        }

        /* Output projection */
        matmul(a->attn_out, a->cat, w->wo[b], T, E, E);

        /* Residual */
        for (int i = 0; i < T * E; i++) a->r1[i] = cur[i] + a->attn_out[i];

        /* RMSNorm + SwiGLU MLP */
        rmsnorm_fwd(a->rm2, a->r1, w->rms2[b], T, E);
        matmul(a->mlp_gate, a->rm2, w->w_gate[b], T, E, M);
        matmul(a->mlp_up, a->rm2, w->w_up[b], T, E, M);
        for (int i = 0; i < T * M; i++)
            a->mlp_swi[i] = siluf(a->mlp_gate[i]) * a->mlp_up[i];
        matmul(a->mlp_out, a->mlp_swi, w->w_down[b], T, M, E);

        /* Residual */
        for (int i = 0; i < T * E; i++) a->r2[i] = a->r1[i] + a->mlp_out[i];

        cur = a->r2;
    }

    /* Final RMSNorm + logits */
    rmsnorm_fwd(a->final_rm, cur, w->rms_f, T, E);
    matmul(a->logits, a->final_rm, w->out_w, T, E, VOCAB);

    /* Apply AML physics to logits */
    for (int t = 0; t < T; t++) {
        apply_destiny_to_logits(a->logits + t * VOCAB, VOCAB);
        apply_suffering_to_logits(a->logits + t * VOCAB, VOCAB);
    }

    /* Cross-entropy loss */
    if (!targets) return 0;
    float loss = 0;
    for (int t = 0; t < T; t++) {
        float *lg = a->logits + t * VOCAB;
        row_softmax(lg, VOCAB);  /* softmax for loss calculation */
        float p = lg[targets[t]];
        if (p < 1e-10f) p = 1e-10f;
        loss -= logf(p);
    }
    return loss / T;
}

/* ═══════════════════════════════════════════════════════════════════
 * BACKWARD PASS — gradient computation
 * ═══════════════════════════════════════════════════════════════════ */

static void backward(Ptrs *w, Ptrs *g, Acts *a, int *tokens, int *targets, int T) {
    int E = DIM, H = HEADS, D = HEAD_DIM, M = MLP_DIM;
    float *d_logits = (float *)calloc(T * VOCAB, sizeof(float));
    float *d_final  = (float *)calloc(T * E, sizeof(float));
    float *d_cur    = (float *)calloc(T * E, sizeof(float));

    /* d_logits = softmax(logits) - one_hot(target) */
    for (int t = 0; t < T; t++) {
        float *lg = a->logits + t * VOCAB;
        /* logits already contain softmax probs from forward */
        for (int v = 0; v < VOCAB; v++) d_logits[t*VOCAB+v] = lg[v];
        d_logits[t*VOCAB+targets[t]] -= 1.0f;
        for (int v = 0; v < VOCAB; v++) d_logits[t*VOCAB+v] /= T;
    }

    /* d_out_w: [E, V] <- d_logits^T @ final_rm */
    matmul_atb(g->out_w, a->final_rm, d_logits, E, T, VOCAB);
    /* d_final_rm: d_logits @ out_w^T */
    matmul_abt(d_final, d_logits, w->out_w, T, VOCAB, E);

    /* Approx RMSNorm backward */
    float *cur_for_rms = (BLOCKS > 0) ? a->r2 : a->x;
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int e = 0; e < E; e++) ss += cur_for_rms[t*E+e] * cur_for_rms[t*E+e];
        float inv = 1.0f / sqrtf(ss / E + 1e-5f);
        for (int e = 0; e < E; e++) {
            d_cur[t*E+e] = d_final[t*E+e] * w->rms_f[e] * inv;
            g->rms_f[e] += d_final[t*E+e] * cur_for_rms[t*E+e] * inv;
        }
    }

    /* Backprop through blocks (reverse) */
    for (int b = BLOCKS - 1; b >= 0; b--) {
        /* d_cur flows through residual to both MLP and attention paths */

        /* MLP backward: d_mlp_out = d_cur (from residual) */
        float *d_mlp = (float *)calloc(T * E, sizeof(float));
        memcpy(d_mlp, d_cur, T * E * sizeof(float));

        /* d_w_down: [M,E] += mlp_swi^T @ d_mlp */
        matmul_atb(g->w_down[b], a->mlp_swi, d_mlp, M, T, E);

        /* d_mlp_swi: d_mlp @ w_down^T [T,M] */
        float *d_swi = (float *)calloc(T * M, sizeof(float));
        matmul_abt(d_swi, d_mlp, w->w_down[b], T, E, M);

        /* SwiGLU backward */
        float *d_gate = (float *)calloc(T * M, sizeof(float));
        float *d_up   = (float *)calloc(T * M, sizeof(float));
        for (int i = 0; i < T * M; i++) {
            d_up[i] = d_swi[i] * siluf(a->mlp_gate[i]);
            d_gate[i] = d_swi[i] * a->mlp_up[i] * siluf_grad(a->mlp_gate[i]);
        }

        /* d_w_gate, d_w_up */
        matmul_atb(g->w_gate[b], a->rm2, d_gate, E, T, M);
        matmul_atb(g->w_up[b], a->rm2, d_up, E, T, M);

        /* d_rm2 */
        float *d_rm2 = (float *)calloc(T * E, sizeof(float));
        matmul_abt(d_rm2, d_gate, w->w_gate[b], T, M, E);
        float *tmp = (float *)calloc(T * E, sizeof(float));
        matmul_abt(tmp, d_up, w->w_up[b], T, M, E);
        for (int i = 0; i < T * E; i++) d_rm2[i] += tmp[i];

        /* RMSNorm2 backward (approx) */
        for (int t = 0; t < T; t++) {
            float ss = 0;
            for (int e = 0; e < E; e++) ss += a->r1[t*E+e] * a->r1[t*E+e];
            float inv = 1.0f / sqrtf(ss / E + 1e-5f);
            for (int e = 0; e < E; e++) {
                d_cur[t*E+e] += d_rm2[t*E+e] * w->rms2[b][e] * inv;
                g->rms2[b][e] += d_rm2[t*E+e] * a->r1[t*E+e] * inv;
            }
        }

        /* Attention backward: d_attn_out = d_cur (from residual to r1) */
        float *d_attn = (float *)calloc(T * E, sizeof(float));
        memcpy(d_attn, d_cur, T * E * sizeof(float));

        /* d_wo: [E,E] += cat^T @ d_attn */
        matmul_atb(g->wo[b], a->cat, d_attn, E, T, E);

        /* d_cat: d_attn @ wo^T */
        float *d_cat = (float *)calloc(T * E, sizeof(float));
        matmul_abt(d_cat, d_attn, w->wo[b], T, E, E);

        /* Distribute d_cat to per-head QKV gradients (simplified) */
        for (int h = 0; h < H; h++) {
            float gate_logits[3] = { w->gate[b][h*3+0],
                                     w->gate[b][h*3+1],
                                     w->gate[b][h*3+2] };
            row_softmax(gate_logits, 3);
            /* Gradient for wq, wk, wv (standard) — scaled by gate[0] */
            float ga = gate_logits[0];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) {
                    float dc = d_cat[t*E + h*D + d] * ga;
                    /* Approximate: just push gradient to wv */
                    for (int e = 0; e < E; e++)
                        g->wv[b][h*E*D + e*D + d] += dc * a->rm1[t*E+e] / T;
                }
            /* Gradient for wr (RRPRAM) — scaled by gate[1] */
            float gb = gate_logits[1];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) {
                    float dc = d_cat[t*E + h*D + d] * gb;
                    for (int e = 0; e < E; e++)
                        g->wvr[b][h*E*D + e*D + d] += dc * a->rm1[t*E+e] / T;
                }
        }

        /* RMSNorm1 backward (approx) */
        float *input = (b == 0) ? a->x : a->r2;
        for (int t = 0; t < T; t++) {
            float ss = 0;
            for (int e = 0; e < E; e++) ss += input[t*E+e] * input[t*E+e];
            float inv = 1.0f / sqrtf(ss / E + 1e-5f);
            for (int e = 0; e < E; e++)
                g->rms1[b][e] += d_cur[t*E+e] * input[t*E+e] * inv;
        }

        /* Embedding gradient */
        if (b == 0) {
            for (int t = 0; t < T; t++) {
                for (int e = 0; e < E; e++) {
                    g->tok_emb[tokens[t]*E+e] += d_cur[t*E+e];
                    g->pos_emb[t*E+e] += d_cur[t*E+e];
                }
            }
        }

        free(d_mlp); free(d_swi); free(d_gate); free(d_up);
        free(d_rm2); free(tmp); free(d_attn); free(d_cat);
    }

    free(d_logits); free(d_final); free(d_cur);
}

/* ═══════════════════════════════════════════════════════════════════
 * DARIO EQUATION — 7-force generation (adapted for char-level)
 *
 * p(x|Φ,C) = softmax(
 *   (B + α·H + β·F + γ·A + T) / (τ · temperature)
 * )
 *
 * B = Sequential Chain (char bigram scores)
 * H = Hebbian Resonance (co-occurrence)
 * F = Prophecy Fulfillment (based on prophecy debt)
 * A = Destiny Attraction
 * T = Trauma Gravity
 * ═══════════════════════════════════════════════════════════════════ */

static float char_bigrams[VOCAB][VOCAB];
static float char_freq[VOCAB];
static float dario_alpha = 3.0f, dario_beta = 2.0f, dario_gamma = 1.5f;

static void dario_update_bigrams(int prev, int cur) {
    if (prev >= 0 && prev < VOCAB && cur >= 0 && cur < VOCAB) {
        char_bigrams[prev][cur] += 1.0f;
        char_freq[cur] += 1.0f;
    }
}

static void dario_overlay(float *logits, int last_char, int n) {
    if (last_char < 0 || last_char >= VOCAB) return;
    float alpha_mod = 1.0f + 0.3f * chambers[CH_LOVE] - 0.2f * chambers[CH_RAGE];
    float gamma_mod = 1.0f + 0.4f * chambers[CH_VOID] + 0.2f * chambers[CH_COMPLEX];

    /* B: bigram signal */
    float *B = char_bigrams[last_char];
    float b_max = 0;
    for (int i = 0; i < n; i++) if (B[i] > b_max) b_max = B[i];
    if (b_max < 1e-6f) b_max = 1.0f;

    /* H: Hebbian (frequency-based co-occurrence) */
    float total_freq = 0;
    for (int i = 0; i < n; i++) total_freq += char_freq[i];
    if (total_freq < 1.0f) total_freq = 1.0f;

    /* SwiGLU gate through resonance */
    float gate = 1.0f / (1.0f + expf(-(AML.resonance - 0.5f) * 4.0f));
    float h_gate = siluf(gate * 2.0f);
    float f_gate = siluf(gate * 1.5f);

    for (int i = 0; i < n; i++) {
        float b_term = 4.0f * B[i] / b_max;
        float h_term = alpha_mod * dario_alpha * (char_freq[i] / total_freq) * h_gate;
        float f_term = dario_beta * AML.prophecy_debt * f_gate;
        float a_term = gamma_mod * dario_gamma * AML.destiny_bias * 0.1f;
        float t_term = (i < 32 && AML.trauma > 0.3f) ?
                       AML.trauma * 2.0f * (1.0f - (float)i / 32.0f) : 0;
        logits[i] += b_term + h_term + f_term + a_term + t_term;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION — char-level sampling with Dario overlay
 * ═══════════════════════════════════════════════════════════════════ */

static int sample_token(float *logits, int n, float temp) {
    /* Apply temperature */
    for (int i = 0; i < n; i++) logits[i] /= (temp + 1e-8f);
    row_softmax(logits, n);
    /* Top-k sampling (k=8) */
    int topk = 8;
    int indices[8];
    float probs[8];
    for (int k = 0; k < topk; k++) { indices[k] = 0; probs[k] = -1e9f; }
    for (int i = 0; i < n; i++) {
        if (logits[i] > probs[topk-1]) {
            probs[topk-1] = logits[i];
            indices[topk-1] = i;
            for (int j = topk-2; j >= 0; j--) {
                if (probs[j+1] > probs[j]) {
                    float tp = probs[j]; probs[j] = probs[j+1]; probs[j+1] = tp;
                    int ti = indices[j]; indices[j] = indices[j+1]; indices[j+1] = ti;
                } else break;
            }
        }
    }
    float sum = 0;
    for (int k = 0; k < topk; k++) { if (probs[k] < 0) probs[k] = 0; sum += probs[k]; }
    if (sum < 1e-10f) return indices[0];
    float r = (float)rand() / RAND_MAX * sum;
    float cum = 0;
    for (int k = 0; k < topk; k++) { cum += probs[k]; if (cum >= r) return indices[k]; }
    return indices[0];
}

static void generate_sentence(Ptrs *w, Acts *a, int *context, int ctx_len,
                              char *out_buf, int max_chars, float temp) {
    int pos = 0;
    int last_char = (ctx_len > 0) ? context[ctx_len - 1] : ' ';

    while (pos < max_chars - 1) {
        int T = ctx_len < MAX_T ? ctx_len : MAX_T;
        int *tok_window = context + (ctx_len > MAX_T ? ctx_len - MAX_T : 0);
        forward(w, a, tok_window, NULL, T);

        /* Get logits for last position */
        float logits[VOCAB];
        memcpy(logits, a->logits + (T-1) * VOCAB, VOCAB * sizeof(float));

        /* Dario overlay */
        dario_overlay(logits, last_char, VOCAB);

        int next = sample_token(logits, VOCAB, temp);

        /* Update prophecy debt */
        float raw_logits[VOCAB];
        memcpy(raw_logits, a->logits + (T-1) * VOCAB, VOCAB * sizeof(float));
        AML.prophecy_debt = 0.9f * AML.prophecy_debt
                          + 0.1f * compute_prophecy_debt(raw_logits, next, VOCAB);

        /* Update bigrams */
        dario_update_bigrams(last_char, next);
        last_char = next;

        out_buf[pos++] = (char)next;

        /* Stop at sentence boundary */
        if (next == '.' || next == '!' || next == '?' || next == '\n') break;
        /* Add to context */
        if (ctx_len < MAX_T * 4) context[ctx_len++] = next;
    }
    out_buf[pos] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════
 * 12 BI-DIRECTIONAL ASSOCIATIVE REASONING STEPS
 *
 * Each step generates a sentence. Steps go forward (future) or
 * backward (past) based on prophecy debt and calendar drift.
 *
 * High prophecy debt → more backward steps (cautious)
 * Low prophecy debt → more forward steps (confident)
 *
 * Wormhole: skip steps if confident (only at sentence boundaries)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char sentence[256];
    int direction;       /* -1 = backward, +1 = forward */
    int step_idx;
    int wormhole_skip;   /* 1 if this step was a wormhole jump */
    float debt_at_step;
    float dissonance_at_step;
} ReasoningStep;

static void associative_reasoning(DualModel *dm, const char *prompt,
                                  ReasoningStep *steps, int *n_steps) {
    Acts a;
    acts_alloc(&a);
    dual_blend(dm);

    /* Tokenize prompt */
    int context[MAX_T * 8];
    int ctx_len = 0;
    for (int i = 0; prompt[i] && ctx_len < MAX_T * 4; i++)
        context[ctx_len++] = (unsigned char)prompt[i];

    /* Determine forward/backward split from calendar + prophecy */
    float cal_d = calendar_dissonance(calendar_days_since_epoch());
    float debt = AML.prophecy_debt;
    int n_backward = (int)(NSTEPS * (0.3f + 0.4f * debt + 0.1f * cal_d));
    int n_forward = NSTEPS - n_backward;
    if (n_backward < 1) n_backward = 1;
    if (n_forward < 1) n_forward = 1;
    if (n_backward + n_forward > NSTEPS) n_backward = NSTEPS - n_forward;

    /* MetaJanus prophecy: internal prediction of where steps will lead */
    float meta_prophecy_entropy = 0.5f + 0.3f * cal_d + 0.2f * debt;
    float temp_base = 0.7f + 0.3f * meta_prophecy_entropy;

    int step_count = 0;

    /* Forward steps (future) */
    for (int s = 0; s < n_forward && step_count < NSTEPS; s++) {
        /* Wormhole check: confident → skip */
        int skip = 0;
        if (AML.prophecy_debt < 0.2f && AML.wormhole > 0.1f) {
            float r = (float)rand() / RAND_MAX;
            if (r < AML.wormhole) {
                skip = 1;
                int jump = 1 + rand() % (AML.tunnel_skip_max < 3 ? 1 : 3);
                s += jump - 1;  /* skip ahead */
            }
        }

        ReasoningStep *rs = &steps[step_count];
        rs->direction = 1;
        rs->step_idx = step_count;
        rs->wormhole_skip = skip;
        rs->debt_at_step = AML.prophecy_debt;
        rs->dissonance_at_step = cal_d;

        float temp = temp_base * (1.0f - 0.02f * s);  /* slightly cooler each step */
        generate_sentence(&dm->bw, &a, context, ctx_len, rs->sentence, SENT_LEN, temp);

        update_chambers(step_count);
        step_count++;
    }

    /* Backward steps (past — higher temperature, more exploratory) */
    /* Reset context to original prompt for backward exploration */
    ctx_len = 0;
    for (int i = 0; prompt[i] && ctx_len < MAX_T * 4; i++)
        context[ctx_len++] = (unsigned char)prompt[i];

    for (int s = 0; s < n_backward && step_count < NSTEPS; s++) {
        ReasoningStep *rs = &steps[step_count];
        rs->direction = -1;
        rs->step_idx = step_count;
        rs->wormhole_skip = 0;
        rs->debt_at_step = AML.prophecy_debt;
        rs->dissonance_at_step = cal_d;

        float temp = temp_base * (1.0f + 0.05f * s);  /* warmer = more exploratory */
        generate_sentence(&dm->bw, &a, context, ctx_len, rs->sentence, SENT_LEN, temp);

        update_chambers(step_count);
        step_count++;
    }

    *n_steps = step_count;

    /* Update prophecy debt based on meta-prophecy accuracy */
    float actual_entropy = 0;
    for (int i = 0; i < step_count; i++)
        actual_entropy += steps[i].debt_at_step;
    actual_entropy /= (step_count > 0 ? step_count : 1);
    float prophecy_error = fabsf(meta_prophecy_entropy - actual_entropy);
    AML.prophecy_debt = clamp01(AML.prophecy_debt + prophecy_error * 0.1f);

    acts_free(&a);
}

static void display_reasoning(ReasoningStep *steps, int n_steps) {
    /* Display: backward steps above (top), forward steps below */
    /* Count each direction */
    int n_back = 0, n_fwd = 0;
    for (int i = 0; i < n_steps; i++) {
        if (steps[i].direction == -1) n_back++;
        else n_fwd++;
    }

    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  JANUS ASSOCIATIVE REASONING — %d steps              ║\n", n_steps);
    printf("║  ↑ backward (past): %d  ↓ forward (future): %d       ║\n", n_back, n_fwd);
    printf("╠═══════════════════════════════════════════════════════╣\n");

    /* Backward steps (top, reversed order) */
    for (int i = n_steps - 1; i >= 0; i--) {
        if (steps[i].direction != -1) continue;
        printf("║ ↑%d%s debt=%.2f │ %s\n",
               steps[i].step_idx,
               steps[i].wormhole_skip ? " ⊕WH" : "    ",
               steps[i].debt_at_step,
               steps[i].sentence);
    }

    printf("╠═══════════════ ● ORIGIN ══════════════════════════════╣\n");

    /* Forward steps (bottom) */
    for (int i = 0; i < n_steps; i++) {
        if (steps[i].direction != 1) continue;
        printf("║ ↓%d%s debt=%.2f │ %s\n",
               steps[i].step_idx,
               steps[i].wormhole_skip ? " ⊕WH" : "    ",
               steps[i].debt_at_step,
               steps[i].sentence);
    }

    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("  calendar_drift=%.4f  dissonance=%.4f  blend=%.3f\n",
           calendar_cumulative_drift(calendar_days_since_epoch()),
           calendar_dissonance(calendar_days_since_epoch()),
           0.0f);  /* will be set properly with dual model */
    printf("  prophecy_debt=%.4f  resonance=%.4f  wormhole=%.4f\n\n",
           AML.prophecy_debt, AML.resonance, AML.wormhole);
}

/* ═══════════════════════════════════════════════════════════════════
 * TRAINING LOOP — with Chuck optimizer
 * ═══════════════════════════════════════════════════════════════════ */

static void train(DualModel *dm, const char *path, int max_steps, float lr) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[janus] cannot open %s\n", path); return; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(fsz);
    fread(data, 1, fsz, f);
    fclose(f);

    Acts a;
    acts_alloc(&a);
    int T = MAX_T;
    int *tokens  = (int *)malloc(T * sizeof(int));
    int *targets = (int *)malloc(T * sizeof(int));

    printf("[janus] corpus: %ld bytes\n", fsz);
    printf("[janus] model A: %d params (%.2fMB)\n",
           dm->A.n_params, dm->A.n_params * 4.0f / 1e6f);
    printf("[janus] model B: %d params (%.2fMB)\n",
           dm->B.n_params, dm->B.n_params * 4.0f / 1e6f);
    printf("[janus] optimizer: Chuck v4 (b1=%.1f b2=%.3f)\n", CHUCK_B1, CHUCK_B2);
    printf("[janus] training: %d steps, lr=%.1e\n", max_steps, lr);

    float best_loss = 1e9f;
    clock_t t0 = clock();

    for (int step = 1; step <= max_steps; step++) {
        /* Alternate training between matrix A and B */
        Model *active = (step % 2 == 0) ? &dm->B : &dm->A;

        int offset = rand() % (fsz - T - 1);
        for (int t = 0; t < T; t++) {
            tokens[t]  = data[offset + t];
            targets[t] = data[offset + t + 1];
        }

        float loss = forward(&active->w, &a, tokens, targets, T);
        backward(&active->w, &active->g, &a, tokens, targets, T);

        /* Chuck optimizer */
        chuck_observe(loss);
        chuck_update(active->data, active->grad, active->chuck_m, active->chuck_v,
                     active->n_params, lr);

        if (loss < best_loss) best_loss = loss;

        if (step % 100 == 0 || step == 1) {
            float elapsed = (float)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  step %5d/%d  loss=%.4f  best=%.4f  chuck=[λ=%.3f S=%.3f σ=%.3f]  %.1f s/s\n",
                   step, max_steps, loss, best_loss,
                   Chuck.dampen, Chuck.lr_scale, Chuck.sigma,
                   step / (elapsed + 1e-6f));
        }
    }

    printf("[janus] training complete. best loss: %.4f\n", best_loss);

    acts_free(&a);
    free(data); free(tokens); free(targets);
}

/* ═══════════════════════════════════════════════════════════════════
 * SAVE / LOAD — binary format
 * ═══════════════════════════════════════════════════════════════════ */

static void save_model(DualModel *dm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[janus] cannot save to %s\n", path); return; }
    int magic = 0x4A414E55; /* "JANU" */
    int version = 1;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&dm->A.n_params, 4, 1, f);
    fwrite(dm->A.data, sizeof(float), dm->A.n_params, f);
    fwrite(dm->B.data, sizeof(float), dm->B.n_params, f);
    /* MetaJanus state */
    fwrite(&MJ, sizeof(MetaJanus), 1, f);
    /* AML state */
    fwrite(&AML, sizeof(AMLState), 1, f);
    /* Kuramoto chambers */
    fwrite(chambers, sizeof(float), NCH, f);
    /* Chuck state */
    fwrite(dm->A.chuck_m, sizeof(float), dm->A.n_params, f);
    fwrite(dm->A.chuck_v, sizeof(float), dm->A.n_params, f);
    fwrite(dm->B.chuck_m, sizeof(float), dm->B.n_params, f);
    fwrite(dm->B.chuck_v, sizeof(float), dm->B.n_params, f);
    long file_size = ftell(f);
    fclose(f);
    printf("[janus] saved to %s (%.2fMB)\n", path, (float)file_size / 1e6f);
}

static int load_model(DualModel *dm, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int magic, version, np;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&np, 4, 1, f);
    if (magic != 0x4A414E55 || np != dm->A.n_params) {
        fprintf(stderr, "[janus] invalid model file\n");
        fclose(f);
        return -1;
    }
    fread(dm->A.data, sizeof(float), dm->A.n_params, f);
    fread(dm->B.data, sizeof(float), dm->B.n_params, f);
    fread(&MJ, sizeof(MetaJanus), 1, f);
    fread(&AML, sizeof(AMLState), 1, f);
    fread(chambers, sizeof(float), NCH, f);
    fread(dm->A.chuck_m, sizeof(float), dm->A.n_params, f);
    fread(dm->A.chuck_v, sizeof(float), dm->A.n_params, f);
    fread(dm->B.chuck_m, sizeof(float), dm->B.n_params, f);
    fread(dm->B.chuck_v, sizeof(float), dm->B.n_params, f);
    fclose(f);
    printf("[janus] loaded from %s\n", path);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * GGUF SPORE EXPORT
 * ═══════════════════════════════════════════════════════════════════ */

#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TENSOR_F32   0

static void gguf_write_kv_string(FILE *f, const char *key, const char *val) {
    uint64_t klen = strlen(key);
    fwrite(&klen, 8, 1, f);
    fwrite(key, 1, klen, f);
    uint32_t type = GGUF_TYPE_STRING;
    fwrite(&type, 4, 1, f);
    uint64_t vlen = strlen(val);
    fwrite(&vlen, 8, 1, f);
    fwrite(val, 1, vlen, f);
}

static void gguf_write_kv_uint32(FILE *f, const char *key, uint32_t val) {
    uint64_t klen = strlen(key);
    fwrite(&klen, 8, 1, f);
    fwrite(key, 1, klen, f);
    uint32_t type = GGUF_TYPE_UINT32;
    fwrite(&type, 4, 1, f);
    fwrite(&val, 4, 1, f);
}

static void gguf_write_kv_float32(FILE *f, const char *key, float val) {
    uint64_t klen = strlen(key);
    fwrite(&klen, 8, 1, f);
    fwrite(key, 1, klen, f);
    uint32_t type = GGUF_TYPE_FLOAT32;
    fwrite(&type, 4, 1, f);
    fwrite(&val, 4, 1, f);
}

static void export_gguf(DualModel *dm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[janus] GGUF export failed\n"); return; }

    uint32_t magic = 0x46475547;  /* "GGUF" */
    uint32_t version = 3;
    uint64_t n_tensors = 2;  /* matrix A + matrix B */
    uint64_t n_kv = 8;

    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&n_tensors, 8, 1, f);
    fwrite(&n_kv, 8, 1, f);

    gguf_write_kv_string(f, "general.architecture", "janus");
    gguf_write_kv_uint32(f, "janus.dim", DIM);
    gguf_write_kv_uint32(f, "janus.heads", HEADS);
    gguf_write_kv_uint32(f, "janus.blocks", BLOCKS);
    gguf_write_kv_uint32(f, "janus.vocab", VOCAB);
    gguf_write_kv_uint32(f, "janus.context", MAX_T);
    gguf_write_kv_float32(f, "janus.calendar_drift",
                          calendar_cumulative_drift(calendar_days_since_epoch()));
    gguf_write_kv_float32(f, "janus.prophecy_debt", AML.prophecy_debt);

    /* Write tensor data */
    fwrite(dm->A.data, sizeof(float), dm->A.n_params, f);
    fwrite(dm->B.data, sizeof(float), dm->B.n_params, f);

    fclose(f);
    printf("[janus] GGUF spore exported to %s\n", path);
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    calendar_init();
    metajanus_init();

    char *train_path = NULL;
    char *load_path = NULL;
    char *save_path = NULL;
    char *gguf_path = NULL;
    char *prompt = NULL;
    int max_steps = 5000;
    float lr = 3e-4f;
    int interactive = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--train") && i+1 < argc) train_path = argv[++i];
        else if (!strcmp(argv[i], "--load") && i+1 < argc) load_path = argv[++i];
        else if (!strcmp(argv[i], "--save") && i+1 < argc) save_path = argv[++i];
        else if (!strcmp(argv[i], "--gguf") && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) max_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--generate") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--interactive")) interactive = 1;
    }

    DualModel dm;
    dual_init(&dm);

    if (load_path) load_model(&dm, load_path);
    if (train_path) train(&dm, train_path, max_steps, lr);
    if (save_path) save_model(&dm, save_path);
    if (gguf_path) export_gguf(&dm, gguf_path);

    if (prompt) {
        ReasoningStep steps[NSTEPS];
        int n_steps = 0;
        associative_reasoning(&dm, prompt, steps, &n_steps);
        display_reasoning(steps, n_steps);
    }

    if (interactive || (!train_path && !prompt)) {
        printf("\n[janus] interactive mode. type a prompt, or 'quit' to exit.\n");
        printf("[janus] calendar_drift=%.4f  dissonance=%.4f\n",
               calendar_cumulative_drift(calendar_days_since_epoch()),
               calendar_dissonance(calendar_days_since_epoch()));
        char buf[1024];
        while (1) {
            printf("\njanus> ");
            if (!fgets(buf, sizeof(buf), stdin)) break;
            buf[strcspn(buf, "\n")] = 0;
            if (!strcmp(buf, "quit") || !strcmp(buf, "exit")) break;
            if (strlen(buf) == 0) continue;

            ReasoningStep steps[NSTEPS];
            int n_steps = 0;
            associative_reasoning(&dm, buf, steps, &n_steps);
            display_reasoning(steps, n_steps);
        }
    }

    dual_free(&dm);
    return 0;
}

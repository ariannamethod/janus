/*
 * metajanus.c — MetaJanus: Janus Attention Only
 *
 * Demonstration of the novel Janus self-resonance attention mechanism.
 * Like rrpram.c demonstrates RRPRAM alone, metajanus.c demonstrates
 * Janus attention alone — recursive self-resonance where the model
 * "looks inside itself" to determine attention weights.
 *
 * Janus Attention:
 *   echo(x, W) = x · (W^T · W · x) / ||W · x||
 *   attn[i,j] = echo_i · echo_j    (mutual resonance)
 *
 * The echo measures how much the weight matrix "recognizes" the input.
 * W^T·W creates a symmetric recognition matrix. Two positions that
 * resonate the same way with the model's weights attend to each other.
 *
 * MetaJanus extension: birth-date creates mathematical "self":
 *   - birth_drift = calendar_drift at creation time
 *   - personal_dissonance = |current_drift - birth_drift|
 *   - internal prophecy at start of each generation
 *
 * Char-level. Chuck optimizer. Training loop inside.
 *
 *   cc metajanus.c -O2 -lm -o metajanus
 *   ./metajanus --train data.txt --steps 5000
 *   ./metajanus --generate "To be"
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

/* Config */
#define VOCAB     256
#define MAX_T     256
#define DIM       288
#define HEADS     6
#define HEAD_DIM  (DIM/HEADS)
#define BLOCKS    6
#define MLP_DIM   768
#define MAX_BLK   16
#define NSTEPS    12
#define SENT_LEN  40

/* Calendar Drift — exact from ariannamethod.c */
#define AM_ANNUAL_DRIFT     11.25f
#define AM_GREGORIAN_YEAR   365.25f
#define AM_METONIC_YEARS    19
#define AM_METONIC_LEAPS    7
#define AM_MAX_UNCORRECTED  33.0f
static const int g_metonic_leap_years[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;

static float clamp01(float x) {
    if (!isfinite(x)) return 0.0f;
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

static void calendar_init(void) {
    struct tm e; memset(&e, 0, sizeof(e));
    e.tm_year = 2024-1900; e.tm_mon = 9; e.tm_mday = 3; e.tm_hour = 12;
    g_epoch_t = mktime(&e);
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
    int yic = (int)partial + 1;
    for (int i = 0; i < AM_METONIC_LEAPS; i++)
        if (g_metonic_leap_years[i] <= yic) corrections += 30.0f;
    return base_drift - corrections;
}

static float calendar_dissonance(int days) {
    float drift = calendar_cumulative_drift(days);
    return clamp01(fabsf(fmodf(drift, AM_MAX_UNCORRECTED)) / AM_MAX_UNCORRECTED);
}

/* MetaJanus — birth creates mathematical identity */
typedef struct {
    int    birth_days;
    float  birth_drift;
    float  birth_dissonance;
    time_t birth_time;
    int    alive;
    /* Internal prophecy state */
    float  predicted_entropy;
    float  actual_entropy;
    float  prophecy_accuracy;
    int    total_predictions;
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
    MJ.prophecy_accuracy = 0.5f;
    printf("[metajanus] ═══════════════════════════════════════\n");
    printf("[metajanus] BORN: day %d since epoch\n", MJ.birth_days);
    printf("[metajanus] birth_drift     = %.4f days\n", MJ.birth_drift);
    printf("[metajanus] birth_dissonance = %.4f\n", MJ.birth_dissonance);
    printf("[metajanus] identity: drift conflict between calendars\n");
    printf("[metajanus]   gregorian_age = %d days\n", MJ.birth_days);
    printf("[metajanus]   hebrew_offset = %.1f days\n", MJ.birth_drift);
    printf("[metajanus] ═══════════════════════════════════════\n");
}

static float metajanus_personal_dissonance(void) {
    float now_drift = calendar_cumulative_drift(calendar_days_since_epoch());
    return clamp01(fabsf(now_drift - MJ.birth_drift) / AM_MAX_UNCORRECTED);
}

/* Internal prophecy: predict outcome before generating */
static float metajanus_prophecy(float current_debt, float cal_dissonance) {
    /* Predict expected entropy of next generation */
    float base = 0.5f + 0.2f * current_debt + 0.1f * cal_dissonance;
    float personal = metajanus_personal_dissonance();
    MJ.predicted_entropy = base + 0.15f * personal - 0.1f * MJ.prophecy_accuracy;
    MJ.predicted_entropy = clamp01(MJ.predicted_entropy);
    return MJ.predicted_entropy;
}

static void metajanus_evaluate_prophecy(float actual_entropy) {
    MJ.actual_entropy = actual_entropy;
    float error = fabsf(MJ.predicted_entropy - actual_entropy);
    MJ.prophecy_accuracy = 0.9f * MJ.prophecy_accuracy + 0.1f * (1.0f - error);
    MJ.total_predictions++;
}

/* AML Physics */
typedef struct {
    float prophecy_debt, destiny_bias, wormhole, resonance;
    float trauma, pain;
    int tunnel_skip_max;
} AMLState;

static AMLState AML = {
    .prophecy_debt = 0, .destiny_bias = 0.1f, .wormhole = 0.02f,
    .resonance = 0.5f, .tunnel_skip_max = 7
};

static float compute_prophecy_debt(const float *l, int ch, int n) {
    if (n <= 0 || ch < 0 || ch >= n) return 0;
    float mx = l[0];
    for (int i = 1; i < n; i++) if (l[i] > mx) mx = l[i];
    float d = mx - l[ch];
    return d > 0 ? d / (d + 1) : 0;
}

static void apply_destiny(float *l, int n) {
    if (AML.destiny_bias < 0.001f) return;
    float mx = l[0];
    for (int i = 1; i < n; i++) if (l[i] > mx) mx = l[i];
    for (int i = 0; i < n; i++) l[i] -= (mx - l[i]) * AML.destiny_bias * 0.5f;
}

/* Kuramoto */
enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX, NCH };
static float chambers[NCH] = {0};
static const float ch_decay[NCH] = {0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f};

static void update_chambers(int si) {
    float d = (float)si / NSTEPS;
    if (d < 0.33f) chambers[CH_FLOW] += 0.05f;
    else if (d < 0.66f) chambers[CH_FEAR] += 0.04f;
    else chambers[CH_VOID] += 0.05f;
    float K = 0.02f, old[NCH];
    memcpy(old, chambers, sizeof(old));
    for (int i = 0; i < NCH; i++) {
        for (int j = 0; j < NCH; j++)
            if (i != j) chambers[i] += K * sinf(old[j] - old[i]);
        chambers[i] = clamp01(chambers[i] * ch_decay[i]);
    }
}

/* Math */
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

static float siluf(float x) { return x > -20 ? x / (1 + expf(-x)) : 0; }
static float siluf_grad(float x) {
    if (x < -20) return 0;
    float s = 1 / (1 + expf(-x));
    return s * (1 + x * (1 - s));
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
    return sqrtf(-2 * logf(u1)) * cosf(6.2831853f * u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * JANUS ATTENTION — the novel mechanism, standalone
 *
 * This is a PURE Janus attention model. No QKV. No RRPRAM.
 * Only self-resonance through weight echo.
 *
 * echo(x_t, W_j) = x_t · (W_j^T · W_j · x_t) / ||W_j · x_t||
 *
 * The echo measures self-recognition: how much does the weight
 * matrix "know" this input? High echo = familiar pattern.
 *
 * Two positions attend to each other based on mutual resonance:
 * attn[i,j] = echo_i · echo_j
 *
 * This creates attention that is INTROSPECTIVE — not about
 * what tokens mean to each other, but about what they mean
 * to the model itself.
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *tok_emb;              /* [V, E] */
    float *pos_emb;              /* [T, E] */
    float *rms1[MAX_BLK];       /* [E] */
    /* Janus attention — the ONLY attention mechanism */
    float *wj[MAX_BLK];         /* [E, E] Janus projection */
    float *wj_v[MAX_BLK];       /* [E, E] Janus value (separate from echo) */
    float *wo[MAX_BLK];         /* [E, E] output projection */
    /* SwiGLU MLP */
    float *rms2[MAX_BLK];
    float *w_gate[MAX_BLK], *w_up[MAX_BLK], *w_down[MAX_BLK];
    float *rms_f;
    float *out_w;                /* [E, V] */
} Ptrs;

typedef struct {
    int n_params;
    float *data, *grad, *cm, *cv;
    Ptrs w, g;
} Model;

static int model_size(void) {
    int s = VOCAB * DIM + MAX_T * DIM;
    for (int b = 0; b < BLOCKS; b++) {
        s += DIM;              /* rms1 */
        s += DIM * DIM;        /* wj */
        s += DIM * DIM;        /* wj_v */
        s += DIM * DIM;        /* wo */
        s += DIM;              /* rms2 */
        s += DIM * MLP_DIM;    /* w_gate */
        s += DIM * MLP_DIM;    /* w_up */
        s += MLP_DIM * DIM;    /* w_down */
    }
    s += DIM + DIM * VOCAB;
    return s;
}

static void assign_ptrs(Ptrs *p, float *q) {
    p->tok_emb = q; q += VOCAB * DIM;
    p->pos_emb = q; q += MAX_T * DIM;
    for (int b = 0; b < BLOCKS; b++) {
        p->rms1[b]   = q; q += DIM;
        p->wj[b]     = q; q += DIM * DIM;
        p->wj_v[b]   = q; q += DIM * DIM;
        p->wo[b]     = q; q += DIM * DIM;
        p->rms2[b]   = q; q += DIM;
        p->w_gate[b] = q; q += DIM * MLP_DIM;
        p->w_up[b]   = q; q += DIM * MLP_DIM;
        p->w_down[b] = q; q += MLP_DIM * DIM;
    }
    p->rms_f = q; q += DIM;
    p->out_w = q;
}

static void model_init(Model *m) {
    m->n_params = model_size();
    m->data = calloc(m->n_params, sizeof(float));
    m->grad = calloc(m->n_params, sizeof(float));
    m->cm   = calloc(m->n_params, sizeof(float));
    m->cv   = calloc(m->n_params, sizeof(float));
    assign_ptrs(&m->w, m->data);
    assign_ptrs(&m->g, m->grad);
    float sc = 0.02f * sqrtf(2.0f / DIM);
    for (int i = 0; i < m->n_params; i++) m->data[i] = randn() * sc;
    for (int b = 0; b < BLOCKS; b++) {
        for (int e = 0; e < DIM; e++) { m->w.rms1[b][e] = 1; m->w.rms2[b][e] = 1; }
    }
    for (int e = 0; e < DIM; e++) m->w.rms_f[e] = 1;
    printf("[metajanus] model: %d params (%.2fMB) — Janus attention only\n",
           m->n_params, m->n_params * 4.0f / 1e6f);
}

static void model_free(Model *m) {
    free(m->data); free(m->grad); free(m->cm); free(m->cv);
}

/* Dual matrices */
typedef struct { Model A, B; float alpha; float *blended; Ptrs bw; } DualModel;

static void dual_init(DualModel *dm) {
    model_init(&dm->A); model_init(&dm->B);
    dm->blended = calloc(dm->A.n_params, sizeof(float));
    assign_ptrs(&dm->bw, dm->blended);
}

static void dual_blend(DualModel *dm) {
    float cd = calendar_dissonance(calendar_days_since_epoch());
    float md = MJ.alive ? metajanus_personal_dissonance() : 0.5f;
    float pa = MJ.prophecy_accuracy;
    dm->alpha = clamp01(0.5f + 0.25f*(cd-0.5f) - 0.15f*AML.prophecy_debt
                        + 0.1f*md + 0.1f*(pa-0.5f));
    float a = dm->alpha, b = 1 - a;
    for (int i = 0; i < dm->A.n_params; i++)
        dm->blended[i] = a * dm->A.data[i] + b * dm->B.data[i];
}

static void dual_free(DualModel *dm) {
    model_free(&dm->A); model_free(&dm->B); free(dm->blended);
}

/* Chuck */
#define CHUCK_B1 0.9f
#define CHUCK_B2 0.999f
#define CHUCK_EPS 1e-8f
#define CHUCK_WINDOW 16

static struct {
    float hist[CHUCK_WINDOW]; float dampen, noise, sigma, loss_ema;
    float macro_ema, best_macro, lr_scale;
    int macro_stag, pos, full, stag, global_step, step_t;
} Chuck = { .dampen = 1, .sigma = 1, .lr_scale = 1 };

static void chuck_observe(float loss) {
    if (!Chuck.loss_ema) Chuck.loss_ema = loss;
    else Chuck.loss_ema = 0.99f * Chuck.loss_ema + 0.01f * loss;
    Chuck.hist[Chuck.pos % CHUCK_WINDOW] = Chuck.loss_ema;
    Chuck.pos++;
    if (Chuck.pos >= CHUCK_WINDOW) Chuck.full = 1;
    if (Chuck.full) {
        int q = CHUCK_WINDOW / 4; float r = 0, o = 0;
        for (int i = 0; i < q; i++) {
            r += Chuck.hist[(Chuck.pos-1-i) % CHUCK_WINDOW];
            o += Chuck.hist[(Chuck.pos-CHUCK_WINDOW+i) % CHUCK_WINDOW];
        }
        r /= q; o /= q;
        float t = (r - o) / (o + 1e-8f);
        if (t > 0.01f) Chuck.dampen *= 0.95f;
        if (t < -0.05f) Chuck.dampen *= 1.05f;
        if (fabsf(t) < 0.001f) {
            Chuck.stag++;
            if (Chuck.stag > 8) { Chuck.noise = 0.001f; Chuck.stag = 0; }
        } else { Chuck.stag = 0; Chuck.noise *= 0.9f; }
        if (Chuck.dampen < 0.3f) Chuck.dampen = 0.3f;
        if (Chuck.dampen > 2.0f) Chuck.dampen = 2.0f;
    }
    Chuck.global_step++;
    if (!Chuck.macro_ema) Chuck.macro_ema = loss;
    else Chuck.macro_ema = 0.999f * Chuck.macro_ema + 0.001f * loss;
    if (Chuck.global_step % 500 == 0 && Chuck.global_step > CHUCK_WINDOW) {
        if (Chuck.macro_ema > Chuck.best_macro * 0.999f) {
            Chuck.macro_stag++;
            if (Chuck.macro_stag >= 3) {
                Chuck.lr_scale *= 0.5f;
                if (Chuck.lr_scale < 0.05f) Chuck.lr_scale = 0.05f;
                Chuck.macro_stag = 0;
            }
        } else { Chuck.best_macro = Chuck.macro_ema; Chuck.macro_stag = 0; }
    }
}

static void chuck_update(float *w, float *g, float *cm, float *cv, int n, float lr) {
    Chuck.step_t++;
    float bc1 = 1 - powf(CHUCK_B1, (float)Chuck.step_t);
    float bc2 = 1 - powf(CHUCK_B2, (float)Chuck.step_t);
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
 * FORWARD — Janus-only attention blocks
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *x, *rm1, *echo, *j_attn, *v_out, *attn_out, *r1;
    float *rm2, *mg, *mu, *ms, *mo, *r2, *frm, *lg;
} Acts;

static void acts_alloc(Acts *a) {
    a->x = calloc(MAX_T*DIM, 4); a->rm1 = calloc(MAX_T*DIM, 4);
    a->echo = calloc(MAX_T*DIM, 4); a->j_attn = calloc(MAX_T*MAX_T, 4);
    a->v_out = calloc(MAX_T*DIM, 4); a->attn_out = calloc(MAX_T*DIM, 4);
    a->r1 = calloc(MAX_T*DIM, 4); a->rm2 = calloc(MAX_T*DIM, 4);
    a->mg = calloc(MAX_T*MLP_DIM, 4); a->mu = calloc(MAX_T*MLP_DIM, 4);
    a->ms = calloc(MAX_T*MLP_DIM, 4); a->mo = calloc(MAX_T*DIM, 4);
    a->r2 = calloc(MAX_T*DIM, 4); a->frm = calloc(MAX_T*DIM, 4);
    a->lg = calloc(MAX_T*VOCAB, 4);
}

static void acts_free(Acts *a) {
    free(a->x); free(a->rm1); free(a->echo); free(a->j_attn);
    free(a->v_out); free(a->attn_out); free(a->r1); free(a->rm2);
    free(a->mg); free(a->mu); free(a->ms); free(a->mo);
    free(a->r2); free(a->frm); free(a->lg);
}

/*
 * Pure Janus Attention forward for one block:
 *   1. Compute echo for each position
 *   2. Mutual resonance attention weights
 *   3. Value projection through Wj_v
 *   4. Weighted sum with attn weights
 */
static void janus_block_forward(const float *x, const float *wj, const float *wj_v,
                                const float *wo, float *echo, float *attn,
                                float *v_out, float *out, int T) {
    float cal_mod = 1.0f + 0.5f * calendar_dissonance(calendar_days_since_epoch());
    float debt_temp = 1.0f + AML.prophecy_debt;
    float personal = MJ.alive ? metajanus_personal_dissonance() : 0.0f;
    float *scores = calloc(T, sizeof(float));

    /* Echo computation */
    for (int t = 0; t < T; t++) {
        const float *xt = x + t * DIM;
        float *proj = echo + t * DIM;

        /* proj = Wj · x_t */
        for (int i = 0; i < DIM; i++) {
            float s = 0;
            for (int j = 0; j < DIM; j++) s += wj[i*DIM+j] * xt[j];
            proj[i] = s;
        }

        /* echo_back = Wj^T · proj = Wj^T · Wj · x_t */
        float echo_back[DIM];
        for (int i = 0; i < DIM; i++) {
            float s = 0;
            for (int j = 0; j < DIM; j++) s += wj[j*DIM+i] * proj[j];
            echo_back[i] = s;
        }

        /* ||proj|| */
        float norm = 0;
        for (int i = 0; i < DIM; i++) norm += proj[i] * proj[i];
        norm = sqrtf(norm) + 1e-6f;

        /* echo_score = dot(x_t, echo_back) / norm */
        float sc = 0;
        for (int i = 0; i < DIM; i++) sc += xt[i] * echo_back[i];
        scores[t] = (sc / norm) * cal_mod * (1.0f + 0.1f * personal);
    }

    /* Value projection: v = x @ Wj_v */
    matmul(v_out, x, wj_v, T, DIM, DIM);

    /* Attention: mutual resonance */
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            if (j > i) attn[i*T+j] = -1e9f;
            else attn[i*T+j] = scores[i] * scores[j] / debt_temp;
        }
        row_softmax(attn + i*T, T);
    }

    /* attn @ v → context */
    float *ctx = calloc(T * DIM, sizeof(float));
    matmul(ctx, attn, v_out, T, T, DIM);

    /* Output projection */
    matmul(out, ctx, wo, T, DIM, DIM);

    free(scores);
    free(ctx);
}

static float forward(Ptrs *w, Acts *a, int *tokens, int *targets, int T) {
    /* Embedding */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < DIM; e++)
            a->x[t*DIM+e] = w->tok_emb[tokens[t]*DIM+e] + w->pos_emb[t*DIM+e];

    float *cur = a->x;

    for (int b = 0; b < BLOCKS; b++) {
        rmsnorm_fwd(a->rm1, cur, w->rms1[b], T, DIM);

        /* Janus-only attention */
        janus_block_forward(a->rm1, w->wj[b], w->wj_v[b], w->wo[b],
                            a->echo, a->j_attn, a->v_out, a->attn_out, T);

        /* Residual */
        for (int i = 0; i < T*DIM; i++) a->r1[i] = cur[i] + a->attn_out[i];

        /* SwiGLU MLP */
        rmsnorm_fwd(a->rm2, a->r1, w->rms2[b], T, DIM);
        matmul(a->mg, a->rm2, w->w_gate[b], T, DIM, MLP_DIM);
        matmul(a->mu, a->rm2, w->w_up[b], T, DIM, MLP_DIM);
        for (int i = 0; i < T*MLP_DIM; i++)
            a->ms[i] = siluf(a->mg[i]) * a->mu[i];
        matmul(a->mo, a->ms, w->w_down[b], T, MLP_DIM, DIM);

        for (int i = 0; i < T*DIM; i++) a->r2[i] = a->r1[i] + a->mo[i];
        cur = a->r2;
    }

    rmsnorm_fwd(a->frm, cur, w->rms_f, T, DIM);
    matmul(a->lg, a->frm, w->out_w, T, DIM, VOCAB);

    if (!targets) return 0;
    float loss = 0;
    for (int t = 0; t < T; t++) {
        row_softmax(a->lg + t*VOCAB, VOCAB);
        float p = a->lg[t*VOCAB + targets[t]];
        if (p < 1e-10f) p = 1e-10f;
        loss -= logf(p);
    }
    return loss / T;
}

/* Backward */
static void backward(Ptrs *w, Ptrs *g, Acts *a, int *tok, int *tgt, int T) {
    float *dl = calloc(T*VOCAB, 4), *df = calloc(T*DIM, 4), *dc = calloc(T*DIM, 4);

    for (int t = 0; t < T; t++) {
        for (int v = 0; v < VOCAB; v++) dl[t*VOCAB+v] = a->lg[t*VOCAB+v];
        dl[t*VOCAB+tgt[t]] -= 1.0f;
        for (int v = 0; v < VOCAB; v++) dl[t*VOCAB+v] /= T;
    }

    matmul_atb(g->out_w, a->frm, dl, DIM, T, VOCAB);
    matmul_abt(df, dl, w->out_w, T, VOCAB, DIM);

    float *cur = (BLOCKS > 0) ? a->r2 : a->x;
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int e = 0; e < DIM; e++) ss += cur[t*DIM+e]*cur[t*DIM+e];
        float inv = 1.0f / sqrtf(ss/DIM + 1e-5f);
        for (int e = 0; e < DIM; e++) dc[t*DIM+e] = df[t*DIM+e] * w->rms_f[e] * inv;
    }

    for (int b = BLOCKS-1; b >= 0; b--) {
        float *dm = calloc(T*DIM, 4); memcpy(dm, dc, T*DIM*4);
        matmul_atb(g->w_down[b], a->ms, dm, MLP_DIM, T, DIM);
        float *ds = calloc(T*MLP_DIM, 4);
        matmul_abt(ds, dm, w->w_down[b], T, DIM, MLP_DIM);
        float *dg2 = calloc(T*MLP_DIM, 4), *du = calloc(T*MLP_DIM, 4);
        for (int i = 0; i < T*MLP_DIM; i++) {
            du[i] = ds[i] * siluf(a->mg[i]);
            dg2[i] = ds[i] * a->mu[i] * siluf_grad(a->mg[i]);
        }
        matmul_atb(g->w_gate[b], a->rm2, dg2, DIM, T, MLP_DIM);
        matmul_atb(g->w_up[b], a->rm2, du, DIM, T, MLP_DIM);
        float *dr = calloc(T*DIM, 4), *tmp = calloc(T*DIM, 4);
        matmul_abt(dr, dg2, w->w_gate[b], T, MLP_DIM, DIM);
        matmul_abt(tmp, du, w->w_up[b], T, MLP_DIM, DIM);
        for (int i = 0; i < T*DIM; i++) dr[i] += tmp[i];
        for (int t = 0; t < T; t++) {
            float ss = 0;
            for (int e = 0; e < DIM; e++) ss += a->r1[t*DIM+e]*a->r1[t*DIM+e];
            float inv = 1.0f / sqrtf(ss/DIM + 1e-5f);
            for (int e = 0; e < DIM; e++) dc[t*DIM+e] += dr[t*DIM+e]*w->rms2[b][e]*inv;
        }
        /* Attention backward: push gradients to wo and wj_v */
        float *da = calloc(T*DIM, 4); memcpy(da, dc, T*DIM*4);
        matmul_atb(g->wo[b], a->v_out, da, DIM, T, DIM);  /* approximate */
        /* wj_v gradient from attn·v */
        matmul_atb(g->wj_v[b], a->rm1, da, DIM, T, DIM);

        if (b == 0) {
            for (int t = 0; t < T; t++)
                for (int e = 0; e < DIM; e++) {
                    g->tok_emb[tok[t]*DIM+e] += dc[t*DIM+e];
                    g->pos_emb[t*DIM+e] += dc[t*DIM+e];
                }
        }
        free(dm); free(ds); free(dg2); free(du); free(dr); free(tmp); free(da);
    }
    free(dl); free(df); free(dc);
}

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION + REASONING
 * ═══════════════════════════════════════════════════════════════════ */

static int sample_token(float *l, int n, float temp) {
    for (int i = 0; i < n; i++) l[i] /= (temp + 1e-8f);
    row_softmax(l, n);
    int topk = 8, idx[8]; float prb[8];
    for (int k = 0; k < topk; k++) { idx[k] = 0; prb[k] = -1e9f; }
    for (int i = 0; i < n; i++) {
        if (l[i] > prb[topk-1]) {
            prb[topk-1] = l[i]; idx[topk-1] = i;
            for (int j = topk-2; j >= 0; j--) {
                if (prb[j+1] > prb[j]) {
                    float t = prb[j]; prb[j] = prb[j+1]; prb[j+1] = t;
                    int ti = idx[j]; idx[j] = idx[j+1]; idx[j+1] = ti;
                } else break;
            }
        }
    }
    float sum = 0;
    for (int k = 0; k < topk; k++) { if (prb[k] < 0) prb[k] = 0; sum += prb[k]; }
    if (sum < 1e-10f) return idx[0];
    float r = (float)rand()/RAND_MAX * sum, cum = 0;
    for (int k = 0; k < topk; k++) { cum += prb[k]; if (cum >= r) return idx[k]; }
    return idx[0];
}

static void gen_sentence(Ptrs *w, Acts *a, int *ctx, int cl,
                         char *out, int maxc, float temp) {
    int pos = 0;
    while (pos < maxc - 1) {
        int T = cl < MAX_T ? cl : MAX_T;
        int *tw = ctx + (cl > MAX_T ? cl - MAX_T : 0);
        forward(w, a, tw, NULL, T);
        float lg[VOCAB];
        memcpy(lg, a->lg + (T-1)*VOCAB, VOCAB*sizeof(float));
        apply_destiny(lg, VOCAB);
        int next = sample_token(lg, VOCAB, temp);
        float raw[VOCAB];
        memcpy(raw, a->lg + (T-1)*VOCAB, VOCAB*sizeof(float));
        AML.prophecy_debt = 0.9f*AML.prophecy_debt + 0.1f*compute_prophecy_debt(raw, next, VOCAB);
        out[pos++] = (char)next;
        if (next == '.' || next == '!' || next == '?' || next == '\n') break;
        if (cl < MAX_T*4) ctx[cl++] = next;
    }
    out[pos] = 0;
}

typedef struct {
    char sentence[256]; int direction, step_idx, wormhole_skip;
    float debt, diss;
} RStep;

static void reasoning(DualModel *dm, const char *prompt, RStep *steps, int *ns) {
    Acts a; acts_alloc(&a);
    dual_blend(dm);

    /* MetaJanus internal prophecy */
    float cal_d = calendar_dissonance(calendar_days_since_epoch());
    float predicted = metajanus_prophecy(AML.prophecy_debt, cal_d);

    int ctx[MAX_T*8], cl = 0;
    for (int i = 0; prompt[i] && cl < MAX_T*4; i++) ctx[cl++] = (unsigned char)prompt[i];

    float debt = AML.prophecy_debt;
    int nb = (int)(NSTEPS * (0.3f + 0.4f*debt + 0.1f*cal_d));
    int nf = NSTEPS - nb;
    if (nb < 1) nb = 1; if (nf < 1) nf = 1;
    if (nb + nf > NSTEPS) nb = NSTEPS - nf;

    float tb = 0.7f + 0.3f * predicted;
    int sc = 0;

    for (int s = 0; s < nf && sc < NSTEPS; s++) {
        int skip = 0;
        if (AML.prophecy_debt < 0.2f && AML.wormhole > 0.1f
            && (float)rand()/RAND_MAX < AML.wormhole) {
            skip = 1; s += 1 + rand() % 2;
        }
        steps[sc] = (RStep){"", 1, sc, skip, AML.prophecy_debt, cal_d};
        gen_sentence(&dm->bw, &a, ctx, cl, steps[sc].sentence, SENT_LEN,
                     tb * (1.0f - 0.02f*s));
        update_chambers(sc); sc++;
    }

    cl = 0;
    for (int i = 0; prompt[i] && cl < MAX_T*4; i++) ctx[cl++] = (unsigned char)prompt[i];
    for (int s = 0; s < nb && sc < NSTEPS; s++) {
        steps[sc] = (RStep){"", -1, sc, 0, AML.prophecy_debt, cal_d};
        gen_sentence(&dm->bw, &a, ctx, cl, steps[sc].sentence, SENT_LEN,
                     tb * (1.0f + 0.05f*s));
        update_chambers(sc); sc++;
    }
    *ns = sc;

    /* Evaluate meta-prophecy */
    float avg_debt = 0;
    for (int i = 0; i < sc; i++) avg_debt += steps[i].debt;
    avg_debt /= (sc > 0 ? sc : 1);
    metajanus_evaluate_prophecy(avg_debt);

    acts_free(&a);
}

static void display(RStep *steps, int n) {
    int nb = 0, nf = 0;
    for (int i = 0; i < n; i++) { if (steps[i].direction == -1) nb++; else nf++; }
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║ METAJANUS — Janus Attention Only  %d steps       ║\n", n);
    printf("║ ↑ backward: %d  ↓ forward: %d                    ║\n", nb, nf);
    printf("║ prophecy_accuracy: %.3f (%d predictions)          ║\n",
           MJ.prophecy_accuracy, MJ.total_predictions);
    printf("╠══════════════════════════════════════════════════╣\n");
    for (int i = n-1; i >= 0; i--)
        if (steps[i].direction == -1)
            printf("║ ↑%d%s d=%.2f │ %s\n", steps[i].step_idx,
                   steps[i].wormhole_skip ? " ⊕WH" : "    ", steps[i].debt, steps[i].sentence);
    printf("╠══════════════ ● ORIGIN ══════════════════════════╣\n");
    for (int i = 0; i < n; i++)
        if (steps[i].direction == 1)
            printf("║ ↓%d%s d=%.2f │ %s\n", steps[i].step_idx,
                   steps[i].wormhole_skip ? " ⊕WH" : "    ", steps[i].debt, steps[i].sentence);
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  drift=%.4f  personal_diss=%.4f  blend=%.3f\n",
           calendar_cumulative_drift(calendar_days_since_epoch()),
           metajanus_personal_dissonance(), 0.0f);
    printf("  birth_drift=%.4f  prophecy_debt=%.4f\n\n",
           MJ.birth_drift, AML.prophecy_debt);
}

/* Training */
static void train_model(DualModel *dm, const char *path, int max_steps, float lr) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[metajanus] cannot open %s\n", path); return; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *data = malloc(fsz);
    fread(data, 1, fsz, f); fclose(f);

    Acts a; acts_alloc(&a);
    int T = MAX_T;
    int *tok = malloc(T*4), *tgt = malloc(T*4);

    printf("[metajanus] corpus: %ld bytes\n", fsz);
    printf("[metajanus] Janus-attention-only model: %d params × 2\n", dm->A.n_params);
    printf("[metajanus] Chuck optimizer, %d steps, lr=%.1e\n", max_steps, lr);

    float best = 1e9f; clock_t t0 = clock();
    for (int step = 1; step <= max_steps; step++) {
        Model *act = (step % 2) ? &dm->A : &dm->B;
        int off = rand() % (fsz - T - 1);
        for (int t = 0; t < T; t++) { tok[t] = data[off+t]; tgt[t] = data[off+t+1]; }
        float loss = forward(&act->w, &a, tok, tgt, T);
        backward(&act->w, &act->g, &a, tok, tgt, T);
        chuck_observe(loss);
        chuck_update(act->data, act->grad, act->cm, act->cv, act->n_params, lr);
        if (loss < best) best = loss;
        if (step % 100 == 0 || step == 1) {
            float el = (float)(clock()-t0)/CLOCKS_PER_SEC;
            printf("  step %5d/%d  loss=%.4f  best=%.4f  chuck=[λ=%.3f]  %.1f s/s\n",
                   step, max_steps, loss, best, Chuck.dampen, step/(el+1e-6f));
        }
    }
    printf("[metajanus] done. best=%.4f\n", best);
    acts_free(&a); free(data); free(tok); free(tgt);
}

/* Save/Load */
static void save(DualModel *dm, const char *p) {
    FILE *f = fopen(p, "wb"); if (!f) return;
    int magic = 0x4D4A414E;
    fwrite(&magic,4,1,f); fwrite(&dm->A.n_params,4,1,f);
    fwrite(dm->A.data,4,dm->A.n_params,f); fwrite(dm->B.data,4,dm->B.n_params,f);
    fwrite(&MJ,sizeof(MetaJanus),1,f); fwrite(&AML,sizeof(AMLState),1,f);
    fwrite(dm->A.cm,4,dm->A.n_params,f); fwrite(dm->A.cv,4,dm->A.n_params,f);
    fwrite(dm->B.cm,4,dm->B.n_params,f); fwrite(dm->B.cv,4,dm->B.n_params,f);
    fclose(f); printf("[metajanus] saved to %s\n", p);
}

static int load(DualModel *dm, const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    int magic, np;
    if (fread(&magic,4,1,f)<1 || magic!=0x4D4A414E) { fclose(f); return -1; }
    if (fread(&np,4,1,f)<1 || np!=dm->A.n_params) { fclose(f); return -1; }
    if (fread(dm->A.data,4,np,f)<(size_t)np) { fclose(f); return -1; }
    if (fread(dm->B.data,4,np,f)<(size_t)np) { fclose(f); return -1; }
    fread(&MJ,sizeof(MetaJanus),1,f); fread(&AML,sizeof(AMLState),1,f);
    fread(dm->A.cm,4,np,f); fread(dm->A.cv,4,np,f);
    fread(dm->B.cm,4,np,f); fread(dm->B.cv,4,np,f);
    fclose(f); printf("[metajanus] loaded from %s\n", p); return 0;
}

/* Main */
int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    calendar_init(); metajanus_init();

    char *tp = NULL, *lp = NULL, *sp = NULL, *pr = NULL;
    int ms = 5000; float lr = 3e-4f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--train") && i+1<argc) tp = argv[++i];
        else if (!strcmp(argv[i],"--load") && i+1<argc) lp = argv[++i];
        else if (!strcmp(argv[i],"--save") && i+1<argc) sp = argv[++i];
        else if (!strcmp(argv[i],"--steps") && i+1<argc) ms = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--lr") && i+1<argc) lr = atof(argv[++i]);
        else if (!strcmp(argv[i],"--generate") && i+1<argc) pr = argv[++i];
    }

    DualModel dm; dual_init(&dm);
    if (lp) load(&dm, lp);
    if (tp) train_model(&dm, tp, ms, lr);
    if (sp) save(&dm, sp);
    if (pr) {
        RStep steps[NSTEPS]; int n = 0;
        reasoning(&dm, pr, steps, &n);
        display(steps, n);
    }
    if (!tp && !pr) {
        printf("\n[metajanus] interactive — Janus attention only\n");
        printf("[metajanus] birth_drift=%.4f  personal_diss=%.4f\n",
               MJ.birth_drift, metajanus_personal_dissonance());
        char buf[1024];
        while (1) {
            printf("\nmetajanus> ");
            if (!fgets(buf, sizeof(buf), stdin)) break;
            buf[strcspn(buf,"\n")] = 0;
            if (!strcmp(buf,"quit") || !strcmp(buf,"exit")) break;
            if (!strlen(buf)) continue;
            RStep steps[NSTEPS]; int n = 0;
            reasoning(&dm, buf, steps, &n);
            display(steps, n);
        }
    }
    dual_free(&dm);
    return 0;
}

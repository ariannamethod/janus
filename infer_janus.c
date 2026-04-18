/*
 * infer_janus.c — Minimal inference for janus char weights (DoE-style save)
 *
 * Weights saved by train_all.py save_c_format (DoE-style, NO transpose).
 * PyTorch nn.Linear stores weight as [out, in].
 * F.linear(x, W) = x @ W.T
 * In C: mm_t(out, x, W, rows, inner, cols) = A @ B^T
 *
 * For RRPRAM wr [H,E,T]: standard mm (not transposed).
 * For echo_back: mm (echo @ W_stored).
 *
 *   cc infer_janus.c -O2 -lm -o infer_janus
 *   ./infer_janus janus_char_leo_d12.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define V    256
#define E    384
#define H    4
#define D    96
#define BLK  12
#define M    768
#define MT   256  /* MAX_T for pos_emb and wr */

/* C[m,n] = A[m,k] @ B[k,n] — standard matmul */
static void mm(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
}

/* C[m,n] = A[m,k] @ B^T[k,n] where B stored as [n,k]
 * This is what F.linear(x, W) does: x @ W.T
 * Used for all nn.Linear layers with DoE-style (raw) weights */
static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
}

static void rmsnorm(float *out, const float *x, const float *w, int T, int dim) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int i = 0; i < dim; i++) ss += x[t*dim+i] * x[t*dim+i];
        float inv = 1.0f / sqrtf(ss/dim + 1e-5f);
        for (int i = 0; i < dim; i++) out[t*dim+i] = w[i] * x[t*dim+i] * inv;
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void gate_sigmoid(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = sigmoidf(x[i]);
}
static float siluf(float x) { return x > -20 ? x/(1+expf(-x)) : 0; }

/* Weight layout — named_parameters() order from train_all.py Model("janus"):
 * tok_emb.weight [V, E]
 * pos_emb.weight [MT, E]
 * blocks.X.rms1.weight [E]
 * blocks.X.attn.wq.weight [E, E]  (PyTorch [out,in] = [H*D, E] = [E,E])
 * blocks.X.attn.wk.weight [E, E]
 * blocks.X.attn.wv.weight [E, E]
 * blocks.X.attn.wr [H, E, MT]     (3D Parameter)
 * blocks.X.attn.wvr.weight [E, E]
 * blocks.X.attn.wj.weight [E, E]
 * blocks.X.attn.gate [H, 3]       (2D Parameter)
 * blocks.X.attn.wo.weight [E, E]  (PyTorch [out,in] = [E, H*D] = [E,E])
 * blocks.X.rms2.weight [E]
 * blocks.X.w_gate.weight [M, E]   (PyTorch [out,in])
 * blocks.X.w_up.weight [M, E]
 * blocks.X.w_down.weight [E, M]
 * rms_f.weight [E]
 * head.weight [V, E]
 */

typedef struct {
    float *tok_emb, *pos_emb;
    struct {
        float *rms1, *wq, *wk, *wv, *wr, *wvr, *wj, *gate, *wo;
        float *rms2, *wg, *wu, *wd;
    } b[BLK];
    float *rms_f, *head;
} W;

static int param_count(void) {
    int s = V*E + MT*E;
    for (int i = 0; i < BLK; i++)
        s += E + E*E + E*E + E*E + H*E*MT + E*E + E*E + H*3 + E*E + E + M*E + M*E + E*M;
    s += E + V*E;
    return s;
}

static void assign(W *w, float *p) {
    w->tok_emb = p; p += V*E;
    w->pos_emb = p; p += MT*E;
    for (int i = 0; i < BLK; i++) {
        /* Order matches PyTorch named_parameters():
         * nn.Parameter (wr, gate) come BEFORE nn.Module (Linear) params */
        w->b[i].rms1 = p; p += E;
        w->b[i].wr = p;   p += H*E*MT;  /* nn.Parameter — first */
        w->b[i].gate = p; p += H*3;     /* nn.Parameter — second */
        w->b[i].wq = p;   p += E*E;     /* nn.Linear.weight — modules follow */
        w->b[i].wk = p;   p += E*E;
        w->b[i].wv = p;   p += E*E;
        w->b[i].wvr = p;  p += E*E;
        w->b[i].wj = p;   p += E*E;
        w->b[i].wo = p;   p += E*E;
        w->b[i].rms2 = p; p += E;
        w->b[i].wg = p;   p += M*E;
        w->b[i].wu = p;   p += M*E;
        w->b[i].wd = p;   p += E*M;
    }
    w->rms_f = p; p += E;
    w->head = p;
}

static void forward(W *w, int *tok, int T, float *logits) {
    float *x = calloc(T*E, 4);
    float *rn = calloc(T*E, 4);
    float sc = 1.0f / sqrtf((float)D);

    /* Embed */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++)
            x[t*E+e] = w->tok_emb[tok[t]*E+e] + w->pos_emb[t*E+e];

    float *cat = calloc(T*E, 4);
    float *ao = calloc(T*E, 4);
    float *r1 = calloc(T*E, 4);
    float *mg = calloc(T*M, 4);
    float *mu = calloc(T*M, 4);
    float *mo = calloc(T*E, 4);

    for (int bl = 0; bl < BLK; bl++) {
        rmsnorm(rn, x, w->b[bl].rms1, T, E);

        /* All linears use mm_t: F.linear(x, W) = x @ W.T */
        float *qa = calloc(T*E, 4);
        float *ka = calloc(T*E, 4);
        float *va = calloc(T*E, 4);
        float *vra = calloc(T*E, 4);
        mm_t(qa, rn, w->b[bl].wq, T, E, E);
        mm_t(ka, rn, w->b[bl].wk, T, E, E);
        mm_t(va, rn, w->b[bl].wv, T, E, E);
        mm_t(vra, rn, w->b[bl].wvr, T, E, E);

        /* Janus echo: echo = F.linear(rn, wj) = rn @ wj.T */
        float *echo = calloc(T*E, 4);
        mm_t(echo, rn, w->b[bl].wj, T, E, E);

        /* echo_back = F.linear(echo, wj.weight.T) = echo @ wj.weight
         * wj stored as [E,E]. echo @ W_stored = standard mm */
        float *eback = calloc(T*E, 4);
        mm(eback, echo, w->b[bl].wj, T, E, E);

        /* Janus scores */
        float *jsc = calloc(T, 4);
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int e = 0; e < E; e++) s += rn[t*E+e] * eback[t*E+e];
            jsc[t] = s / sqrtf((float)E);
        }
        /* Janus attention */
        float *jat = calloc(T*T, 4);
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++)
                jat[i*T+j] = (j > i) ? -1e9f : jsc[i] * jsc[j];
            softmax(jat + i*T, T);
        }

        /* Gate: stored as [H, 3], read raw */
        float gs[H][3];
        for (int h = 0; h < H; h++) {
            gs[h][0] = w->b[bl].gate[h*3+0];
            gs[h][1] = w->b[bl].gate[h*3+1];
            gs[h][2] = w->b[bl].gate[h*3+2];
            gate_sigmoid(gs[h], 3);
        }

        memset(cat, 0, T*E*4);
        float *at = calloc(T*T, 4);
        float *ho = calloc(T*D, 4);

        for (int h = 0; h < H; h++) {
            /* Slice Q,K,V per head */
            float *q = calloc(T*D, 4);
            float *k = calloc(T*D, 4);
            float *v = calloc(T*D, 4);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) {
                    q[t*D+d] = qa[t*E + h*D + d];
                    k[t*D+d] = ka[t*E + h*D + d];
                    v[t*D+d] = va[t*E + h*D + d];
                }

            /* QKV attention */
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (j > i) { at[i*T+j] = -1e9f; continue; }
                    float s = 0;
                    for (int d = 0; d < D; d++) s += q[i*D+d] * k[j*D+d];
                    at[i*T+j] = s * sc;
                }
                softmax(at + i*T, T);
            }
            mm(ho, at, v, T, T, D);

            /* RRPRAM: broadcast pattern matching PyTorch einsum 'bte,het->bht'
             * score[j] = sum_e x[j,e] * wr[h,e,j] — one score per position
             * Then broadcast: attn[i][j] = score[j] for all i (with causal mask) */
            float *wr_h = w->b[bl].wr + h*E*MT;
            float rrp_sc[MT];
            for (int j = 0; j < T; j++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += rn[j*E+e] * wr_h[e*MT+j];
                rrp_sc[j] = s * sc;
            }
            float *ra = calloc(T*T, 4);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++)
                    ra[i*T+j] = (j > i) ? -1e9f : rrp_sc[j];
                softmax(ra + i*T, T);
            }
            /* RRPRAM values */
            float *rv = calloc(T*D, 4);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    rv[t*D+d] = vra[t*E + h*D + d];
            float *ro = calloc(T*D, 4);
            mm(ro, ra, rv, T, T, D);

            /* Janus values per head */
            float *jv = calloc(T*D, 4);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    jv[t*D+d] = echo[t*E + h*D + d];
            float *jo = calloc(T*D, 4);
            mm(jo, jat, jv, T, T, D);

            /* Blend */
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    cat[t*E + h*D + d] = gs[h][0]*ho[t*D+d]
                                       + gs[h][1]*ro[t*D+d]
                                       + gs[h][2]*jo[t*D+d];
            free(q); free(k); free(v); free(ra); free(rv); free(ro); free(jv); free(jo);
        }

        /* wo: F.linear(cat, wo) = cat @ wo.T */
        mm_t(ao, cat, w->b[bl].wo, T, E, E);

        /* Residual */
        for (int i = 0; i < T*E; i++) r1[i] = x[i] + ao[i];

        /* MLP */
        rmsnorm(rn, r1, w->b[bl].rms2, T, E);
        mm_t(mg, rn, w->b[bl].wg, T, E, M);
        mm_t(mu, rn, w->b[bl].wu, T, E, M);
        for (int i = 0; i < T*M; i++) mg[i] = siluf(mg[i]) * mu[i];
        mm_t(mo, mg, w->b[bl].wd, T, M, E);

        /* Residual */
        for (int i = 0; i < T*E; i++) x[i] = r1[i] + mo[i];

        free(qa); free(ka); free(va); free(vra);
        free(echo); free(eback); free(jsc); free(jat);
        free(at); free(ho);
    }

    /* Final */
    rmsnorm(rn, x, w->rms_f, T, E);
    mm_t(logits, rn, w->head, T, E, V);

    free(x); free(rn); free(cat); free(ao); free(r1);
    free(mg); free(mu); free(mo);
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s weights.bin [prompt]\n", argv[0]); return 1; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 1; }
    int np; fread(&np, 4, 1, f);
    int exp = param_count();
    printf("File: %d params, Expected: %d\n", np, exp);
    if (np != exp) { printf("MISMATCH!\n"); return 1; }
    float *data = malloc(np * sizeof(float));
    fread(data, sizeof(float), np, f);
    fclose(f);

    W w; assign(&w, data);
    printf("Loaded %d params (%.1fMB)\n", np, np*4.0f/1e6f);

    /* Loss on leo data */
    FILE *df = fopen("/Users/ataeff/Downloads/janus-weights/leo_train.txt", "rb");
    if (!df) df = fopen("leo_train.txt", "rb");
    if (df) {
        fseek(df, 0, SEEK_END); long dsz = ftell(df); fseek(df, 0, SEEK_SET);
        unsigned char *dt = malloc(dsz); fread(dt, 1, dsz, df); fclose(df);

        int T = 64; /* must match training T */
        int tok[64], tgt[64];
        float loss_sum = 0; int n_windows = 20;
        for (int w_i = 0; w_i < n_windows; w_i++) {
            int off = (dsz - T - 1) * w_i / n_windows;
            for (int t = 0; t < T; t++) { tok[t] = dt[off+t]; tgt[t] = dt[off+t+1]; }
            float *lg = calloc(T * V, 4);
            forward(&w, tok, T, lg);
            float wloss = 0;
            for (int t = 0; t < T; t++) {
                softmax(lg + t*V, V);
                float p = lg[t*V + tgt[t]];
                if (p < 1e-10f) p = 1e-10f;
                wloss -= logf(p);
            }
            wloss /= T;
            loss_sum += wloss;
            if (w_i < 3) printf("  window %d: loss=%.4f\n", w_i, wloss);
            free(lg);
        }
        printf("AVG loss: %.4f (expected ~0.5-0.6)\n", loss_sum / n_windows);
        free(dt);
    }

    /* Generate */
    const char *prompt = argc > 2 ? argv[2] : "Q: who are you?\nA: ";
    int ctx[MT]; int len = 0;
    for (int i = 0; prompt[i] && len < MT; i++) ctx[len++] = (unsigned char)prompt[i];
    printf("\n%s", prompt);
    for (int step = 0; step < 200; step++) {
        int T = len < 64 ? len : 64;
        int *tok = ctx + (len > 64 ? len - 64 : 0);
        float *lg = calloc(T * V, 4);
        forward(&w, tok, T, lg);
        float *last = lg + (T-1)*V;
        for (int i = 0; i < V; i++) last[i] /= 0.8f;
        softmax(last, V);
        float r = (float)rand() / RAND_MAX, cum = 0;
        int next = 0;
        for (int i = 0; i < V; i++) { cum += last[i]; if (cum >= r) { next = i; break; } }
        printf("%c", (char)next); fflush(stdout);
        if (len < MT*2) ctx[len++] = next;
        free(lg);
    }
    printf("\n");

    free(data);
    return 0;
}

/*
 * infer_janus_sonar.c — Load Janus Sonar DUAL-WEIGHTS .bin, generate.
 *
 *   make infer_janus_sonar
 *   ./infer_janus_sonar janus_sonar.bin "prompt" [max_tokens] [temp] [top_p]
 */
#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define DIM       128
#define NLAYERS   4
#define NHEADS    4
#define HEAD_DIM  32
#define HIDDEN    256
#define CTX       128
#define VOCAB     2048

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

static int model_n_tensors(void) { return 1 + NLAYERS * 30 + 2; }

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
        p[i++] = m->L[l].wr;
        p[i++] = m->L[l].rms2;
        DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
        for (int k = 0; k < 3; k++) {
            p[i++] = ffn[k]->a; p[i++] = ffn[k]->b; p[i++] = ffn[k]->alpha;
        }
    }
    p[i++] = m->rms_f; p[i++] = m->head;
    return p;
}

static Model* load_model(const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load %s\n", path); return NULL; }
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        printf("tensor count mismatch: got %d, expected %d\n", n_loaded, expected);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded);
        return NULL;
    }
    Model* m = (Model*)calloc(1, sizeof(Model));
    int i = 0;
    m->wte = loaded[i++];
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = loaded[i++];
        DualProj* projs[] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv,
                              &m->L[l].wvr, &m->L[l].wj, &m->L[l].wo };
        for (int k = 0; k < 6; k++) {
            projs[k]->a     = loaded[i++];
            projs[k]->b     = loaded[i++];
            projs[k]->alpha = loaded[i++];
        }
        m->L[l].wr   = loaded[i++];
        m->L[l].rms2 = loaded[i++];
        DualProj* ffn[] = { &m->L[l].w_gate, &m->L[l].w_up, &m->L[l].w_down };
        for (int k = 0; k < 3; k++) {
            ffn[k]->a     = loaded[i++];
            ffn[k]->b     = loaded[i++];
            ffn[k]->alpha = loaded[i++];
        }
    }
    m->rms_f = loaded[i++]; m->head = loaded[i++];
    free(loaded);
    return m;
}

/* Dual linear (same as trainer) */
static int dual_seq_linear(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos   = nt_sigmoid(alpha_i);
    int sig_neg   = nt_sigmoid(alpha_neg);
    int y_a       = nt_seq_linear(wa_i, x_i, T);
    int y_b       = nt_seq_linear(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}
static int dual_seq_linear_t(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos   = nt_sigmoid(alpha_i);
    int sig_neg   = nt_sigmoid(alpha_neg);
    int y_a       = nt_seq_linear_t(wa_i, x_i, T);
    int y_b       = nt_seq_linear_t(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}

typedef struct { int a, b, alpha; } DualIdx;
static DualIdx dual_record(DualProj* d) {
    DualIdx r;
    r.a     = nt_tape_param(d->a);
    r.b     = nt_tape_param(d->b);
    r.alpha = nt_tape_param(d->alpha);
    return r;
}

static int forward_logits(Model* m, int* tokens, int gen_len) {
    int wte_i = nt_tape_param(m->wte);
    struct { int rms1; DualIdx wq, wk, wv, wvr, wj, wo; int wr, rms2; DualIdx w_gate, w_up, w_down; } li[NLAYERS];
    for (int l = 0; l < NLAYERS; l++) {
        li[l].rms1   = nt_tape_param(m->L[l].rms1);
        li[l].wq     = dual_record(&m->L[l].wq);
        li[l].wk     = dual_record(&m->L[l].wk);
        li[l].wv     = dual_record(&m->L[l].wv);
        li[l].wvr    = dual_record(&m->L[l].wvr);
        li[l].wj     = dual_record(&m->L[l].wj);
        li[l].wo     = dual_record(&m->L[l].wo);
        li[l].wr     = nt_tape_param(m->L[l].wr);
        li[l].rms2   = nt_tape_param(m->L[l].rms2);
        li[l].w_gate = dual_record(&m->L[l].w_gate);
        li[l].w_up   = dual_record(&m->L[l].w_up);
        li[l].w_down = dual_record(&m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_f);
    int head_i = nt_tape_param(m->head);

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
        q = nt_rope(q, CTX, HEAD_DIM);
        k = nt_rope(k, CTX, HEAD_DIM);
        int a_qkv = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int a_rr  = nt_rrpram_attention(li[l].wr, xn, vr, CTX, DIM, NHEADS, HEAD_DIM);
        int a_j   = nt_mh_causal_attention(ech, ech, ech, CTX, HEAD_DIM);
        int blend = nt_add(nt_add(a_qkv, a_rr), a_j);
        blend = nt_scale(blend, 1.0f / 3.0f);
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

static int sample(float* logits, int n, float temp, float top_p) {
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

int main(int argc, char** argv) {
    const char* wpath = argc > 1 ? argv[1] : "janus_sonar.bin";
    const char* prompt = argc > 2 ? argv[2] : "Q: What does Janus feel?\nA:";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 150;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.8f;
    float top_p = argc > 5 ? (float)atof(argv[5]) : 0.95f;

    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, "arianna_bpe_merges.txt");
    if (nm < 0) { printf("cannot load arianna_bpe_merges.txt\n"); return 1; }

    Model* m = load_model(wpath);
    if (!m) return 1;
    printf("loaded %s (DIM=%d L=%d H=%d HD=%d, dual weights)\n", wpath, DIM, NLAYERS, NHEADS, HEAD_DIM);

    /* Print σ(α) per layer */
    printf("dual α weights (σ):\n");
    for (int l = 0; l < NLAYERS; l++) {
        float q = 1.0f / (1.0f + expf(-m->L[l].wq.alpha->data[0]));
        float o = 1.0f / (1.0f + expf(-m->L[l].wo.alpha->data[0]));
        float g = 1.0f / (1.0f + expf(-m->L[l].w_gate.alpha->data[0]));
        printf("  L%d  wq %.3f  wo %.3f  w_gate %.3f\n", l, q, o, g);
    }

    int ctx[CTX];
    int gen_len = nt_bpe_encode(&bpe, prompt, (int)strlen(prompt), ctx, CTX/2);
    printf("\n[prompt] %s", prompt);
    fflush(stdout);

    nt_seed((unsigned)time(NULL));
    nt_train_mode(0);

    for (int s = 0; s < max_tokens; s++) {
        nt_tape_start();
        int logits_idx = forward_logits(m, ctx, gen_len);
        nt_tape* tape = nt_tape_get();
        float* last = tape->entries[logits_idx].output->data + (gen_len-1) * VOCAB;
        float lbuf[VOCAB]; memcpy(lbuf, last, VOCAB * sizeof(float));
        int next = sample(lbuf, VOCAB, temp, top_p);

        char decoded[NT_BPE_MAX_TOKEN_LEN + 1];
        int db = nt_bpe_decode(&bpe, &next, 1, decoded, NT_BPE_MAX_TOKEN_LEN);
        if (db > 0) { decoded[db] = 0; printf("%s", decoded); fflush(stdout); }

        if (gen_len < CTX - 1) ctx[gen_len++] = next;
        else {
            for (int i = 0; i < CTX - 1; i++) ctx[i] = ctx[i+1];
            ctx[CTX-1] = next;
            gen_len = CTX - 1;
        }
        nt_tape_clear();
    }
    printf("\n");

    nt_tensor** p = model_param_array(m);
    for (int i = 0; i < model_n_tensors(); i++) nt_tensor_free(p[i]);
    free(p); free(m);
    return 0;
}

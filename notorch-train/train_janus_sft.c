/*
 * train_janus_sft.c — LoRA supervised fine-tuning of microjanus.
 *
 * Loads a pretrained microjanus (dual or single), freezes all its weights
 * via nt_tape_freeze_param(), then trains rank-8 low-rank adapters on
 * the three content-attention projections (wq, wk, wv) per layer. Output
 * of each projection becomes:
 *
 *    y = W_base · x + (α/rank) · B · (A · x)
 *
 * with A: [rank, DIM], B: [DIM, rank], α=16 (standard LoRA α=2×rank), and
 * B initialized to zero so the adapter starts with no effect.
 *
 * Only A and B are trainable. This is the δ of θ = ε + γ + αδ for a given
 * voice. Corpus: a chunk of the Leo dataset (Q/A dash dialogue).
 *
 *   make train_janus_sft
 *   ./train_janus_sft ../weights/microjanus_dual_sym_5k.bin leo_sft.txt [steps] [lr]
 */
#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

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

#define LOG_EVERY   50
#define CKPT_EVERY  500
#define SFT_PREFIX  "janus_sft_ckpt"

typedef struct { nt_tensor *a, *b, *alpha; } DualProj;
typedef struct { nt_tensor *A, *B; } LoRA;

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1;
        DualProj wq, wk, wv, wvr, wj, wo;
        nt_tensor *wr, *rms2;
        DualProj w_gate, w_up, w_down;
        /* LoRA adapters on content Q, K, V */
        LoRA lora_q, lora_k, lora_v;
    } L[NLAYERS];
    nt_tensor *rms_f;
    nt_tensor *head;
} Model;

static int base_n_tensors(void) { return 1 + NLAYERS * 30 + 2; }

static nt_tensor** base_param_array(Model* m) {
    int n = base_n_tensors();
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

static int load_base(Model* m, const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) { printf("cannot load %s\n", path); return -1; }
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

/* Init LoRA: A ~ N(0, 1/sqrt(rank)), B = zeros → zero-effect at step 0 */
static void lora_init(LoRA* a, int in_dim, int out_dim) {
    a->A = nt_tensor_new2d(LORA_RANK, in_dim);
    nt_tensor_xavier(a->A, in_dim, LORA_RANK);
    a->B = nt_tensor_new2d(out_dim, LORA_RANK);
    /* zero init B — standard LoRA so adapter starts at identity */
    memset(a->B->data, 0, a->B->len * sizeof(float));
}

static void init_adapters(Model* m) {
    for (int l = 0; l < NLAYERS; l++) {
        lora_init(&m->L[l].lora_q, DIM, DIM);
        lora_init(&m->L[l].lora_k, DIM, DIM);
        lora_init(&m->L[l].lora_v, DIM, DIM);
    }
}

static int sft_n_adapters(void) { return NLAYERS * 3 * 2; }   /* q,k,v × A,B */

static nt_tensor** adapter_param_array(Model* m) {
    int n = sft_n_adapters();
    nt_tensor** p = (nt_tensor**)malloc(n * sizeof(nt_tensor*));
    int i = 0;
    for (int l = 0; l < NLAYERS; l++) {
        p[i++]=m->L[l].lora_q.A; p[i++]=m->L[l].lora_q.B;
        p[i++]=m->L[l].lora_k.A; p[i++]=m->L[l].lora_k.B;
        p[i++]=m->L[l].lora_v.A; p[i++]=m->L[l].lora_v.B;
    }
    return p;
}

static void save_adapters(Model* m, const char* prefix) {
    char path[256]; snprintf(path, sizeof(path), "%s.bin", prefix);
    nt_tensor** p = adapter_param_array(m);
    nt_save(path, p, sft_n_adapters());
    free(p);
}

static void model_free(Model* m) {
    nt_tensor** p = base_param_array(m);
    for (int i = 0; i < base_n_tensors(); i++) nt_tensor_free(p[i]);
    free(p);
    for (int l = 0; l < NLAYERS; l++) {
        nt_tensor_free(m->L[l].lora_q.A); nt_tensor_free(m->L[l].lora_q.B);
        nt_tensor_free(m->L[l].lora_k.A); nt_tensor_free(m->L[l].lora_k.B);
        nt_tensor_free(m->L[l].lora_v.A); nt_tensor_free(m->L[l].lora_v.B);
    }
    free(m);
}

/* Dual linear (base, frozen) + LoRA delta: y = base + scale · B · (A · x)
   Record shape: 3 base params frozen + 2 LoRA params trainable. */
typedef struct { int a, b, alpha; } DualIdx;

static DualIdx dual_record_frozen(DualProj* d) {
    DualIdx r;
    r.a     = nt_tape_param(d->a);     nt_tape_freeze_param(r.a);
    r.b     = nt_tape_param(d->b);     nt_tape_freeze_param(r.b);
    r.alpha = nt_tape_param(d->alpha); nt_tape_freeze_param(r.alpha);
    return r;
}

static int dual_plus_lora(int wa_i, int wb_i, int alpha_i,
                          int lora_A_i, int lora_B_i,
                          int x_i, int T) {
    /* Base dual: σ(α)·(W_A·x) + σ(−α)·(W_B·x) */
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear(wa_i, x_i, T), y_b = nt_seq_linear(wb_i, x_i, T);
    int base = nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));

    /* LoRA delta: scale · B · (A · x). A: [rank, DIM], B: [DIM, rank] */
    int mid   = nt_seq_linear(lora_A_i, x_i, T);    /* [T, rank] */
    int delta = nt_seq_linear(lora_B_i, mid, T);    /* [T, DIM] */
    delta     = nt_scale(delta, LORA_SCALE);

    return nt_add(base, delta);
}

static int dual_seq_linear_frozen(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear(wa_i, x_i, T), y_b = nt_seq_linear(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}

static int dual_seq_linear_t_frozen(int wa_i, int wb_i, int alpha_i, int x_i, int T) {
    int alpha_neg = nt_scale(alpha_i, -1.0f);
    int sig_pos = nt_sigmoid(alpha_i), sig_neg = nt_sigmoid(alpha_neg);
    int y_a = nt_seq_linear_t(wa_i, x_i, T), y_b = nt_seq_linear_t(wb_i, x_i, T);
    return nt_add(nt_scale_by_t(y_a, sig_pos), nt_scale_by_t(y_b, sig_neg));
}

static int forward(Model* m, int* tokens, int* targets) {
    /* wte, rms_f, head — frozen */
    int wte_i = nt_tape_param(m->wte); nt_tape_freeze_param(wte_i); nt_tape_no_decay(wte_i);
    struct {
        int rms1;
        DualIdx wq, wk, wv, wvr, wj, wo;
        int wr, rms2;
        DualIdx w_gate, w_up, w_down;
        int lora_q_A, lora_q_B, lora_k_A, lora_k_B, lora_v_A, lora_v_B;
    } li[NLAYERS];

    for (int l = 0; l < NLAYERS; l++) {
        li[l].rms1 = nt_tape_param(m->L[l].rms1); nt_tape_freeze_param(li[l].rms1);
        li[l].wq  = dual_record_frozen(&m->L[l].wq);
        li[l].wk  = dual_record_frozen(&m->L[l].wk);
        li[l].wv  = dual_record_frozen(&m->L[l].wv);
        li[l].wvr = dual_record_frozen(&m->L[l].wvr);
        li[l].wj  = dual_record_frozen(&m->L[l].wj);
        li[l].wo  = dual_record_frozen(&m->L[l].wo);
        li[l].wr  = nt_tape_param(m->L[l].wr);  nt_tape_freeze_param(li[l].wr);
        li[l].rms2 = nt_tape_param(m->L[l].rms2); nt_tape_freeze_param(li[l].rms2);
        li[l].w_gate = dual_record_frozen(&m->L[l].w_gate);
        li[l].w_up   = dual_record_frozen(&m->L[l].w_up);
        li[l].w_down = dual_record_frozen(&m->L[l].w_down);
        /* LoRA — TRAINABLE */
        li[l].lora_q_A = nt_tape_param(m->L[l].lora_q.A);
        li[l].lora_q_B = nt_tape_param(m->L[l].lora_q.B);
        li[l].lora_k_A = nt_tape_param(m->L[l].lora_k.A);
        li[l].lora_k_B = nt_tape_param(m->L[l].lora_k.B);
        li[l].lora_v_A = nt_tape_param(m->L[l].lora_v.A);
        li[l].lora_v_B = nt_tape_param(m->L[l].lora_v.B);
    }
    int rmsf_i = nt_tape_param(m->rms_f); nt_tape_freeze_param(rmsf_i);
    int head_i = nt_tape_param(m->head);  nt_tape_freeze_param(head_i);

    nt_tensor* tok_t = nt_tensor_new(CTX);
    nt_tensor* tgt_t = nt_tensor_new(CTX);
    for (int i = 0; i < CTX; i++) { tok_t->data[i] = (float)tokens[i]; tgt_t->data[i] = (float)targets[i]; }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(wte_i, -1, tok_i, CTX, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        int xn = nt_seq_rmsnorm(h, li[l].rms1, CTX, DIM);
        /* Q, K, V with LoRA adapters */
        int q   = dual_plus_lora(li[l].wq.a, li[l].wq.b, li[l].wq.alpha,
                                 li[l].lora_q_A, li[l].lora_q_B, xn, CTX);
        int k   = dual_plus_lora(li[l].wk.a, li[l].wk.b, li[l].wk.alpha,
                                 li[l].lora_k_A, li[l].lora_k_B, xn, CTX);
        int v   = dual_plus_lora(li[l].wv.a, li[l].wv.b, li[l].wv.alpha,
                                 li[l].lora_v_A, li[l].lora_v_B, xn, CTX);
        /* vr, echo — frozen, no adapter */
        int vr  = dual_seq_linear_frozen  (li[l].wvr.a, li[l].wvr.b, li[l].wvr.alpha, xn, CTX);
        int ech = dual_seq_linear_t_frozen(li[l].wj.a,  li[l].wj.b,  li[l].wj.alpha,  xn, CTX);
        q = nt_rope(q, CTX, HEAD_DIM); k = nt_rope(k, CTX, HEAD_DIM);
        int a_qkv = nt_mh_causal_attention(q, k, v, CTX, HEAD_DIM);
        int a_rr  = nt_rrpram_attention(li[l].wr, xn, vr, CTX, DIM, NHEADS, HEAD_DIM);
        int a_j   = nt_mh_causal_attention(ech, ech, ech, CTX, HEAD_DIM);
        int blend = nt_scale(nt_add(nt_add(a_qkv, a_rr), a_j), 1.0f / 3.0f);
        int proj = dual_seq_linear_frozen(li[l].wo.a, li[l].wo.b, li[l].wo.alpha, blend, CTX);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l].rms2, CTX, DIM);
        int g = nt_silu(dual_seq_linear_frozen(li[l].w_gate.a, li[l].w_gate.b, li[l].w_gate.alpha, xn, CTX));
        int u =         dual_seq_linear_frozen(li[l].w_up.a,   li[l].w_up.b,   li[l].w_up.alpha,   xn, CTX);
        int d =         dual_seq_linear_frozen(li[l].w_down.a, li[l].w_down.b, li[l].w_down.alpha, nt_mul(g, u), CTX);
        h = nt_add(h, d);
    }
    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, VOCAB);
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0+tv.tv_usec/1000.0; }

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s base_weights.bin corpus.txt [steps] [lr]\n", argv[0]);
        return 1;
    }
    const char* wpath = argv[1];
    const char* cpath = argv[2];
    int steps = argc > 3 ? atoi(argv[3]) : 1500;
    float base_lr = argc > 4 ? (float)atof(argv[4]) : 1e-3f;

    printf("════════════════════════════════════════════════════════\n");
    printf("  notorch — JANUS SFT (LoRA rank=%d, α=%.0f, scale=%.2f)\n", LORA_RANK, LORA_ALPHA, LORA_SCALE);
    printf("  DIM=%d L=%d H=%d HD=%d CTX=%d V=%d\n", DIM, NLAYERS, NHEADS, HEAD_DIM, CTX, VOCAB);
    printf("  Base frozen, LoRA on wq/wk/wv per layer\n");
    printf("  Chuck optimizer, %d steps, lr=%.1e\n", steps, base_lr);
    printf("════════════════════════════════════════════════════════\n");

    nt_bpe bpe;
    if (nt_bpe_load(&bpe, "arianna_bpe_merges.txt") < 0) {
        printf("cannot load arianna_bpe_merges.txt\n"); return 1;
    }
    printf("bpe: %d merges, vocab %d\n", bpe.n_merges, bpe.vocab_size);

    FILE* f = fopen(cpath, "rb");
    if (!f) { printf("cannot open %s\n", cpath); return 1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char* raw = (char*)malloc(fsize + 1);
    fread(raw, 1, fsize, f); raw[fsize] = 0; fclose(f);
    int* encoded = (int*)malloc(fsize * sizeof(int));
    int n_tokens = nt_bpe_encode(&bpe, raw, (int)fsize, encoded, (int)fsize);
    free(raw);
    printf("corpus: %.1f KB → %d BPE tokens\n", fsize/1024.0, n_tokens);

    nt_seed(42);
    Model* m = (Model*)calloc(1, sizeof(Model));
    if (load_base(m, wpath) < 0) { free(m); return 1; }
    init_adapters(m);

    long np_base = 0, np_adapter = 0;
    nt_tensor** bp = base_param_array(m);
    for (int i = 0; i < base_n_tensors(); i++) np_base += bp[i]->len;
    free(bp);
    nt_tensor** ap = adapter_param_array(m);
    for (int i = 0; i < sft_n_adapters(); i++) np_adapter += ap[i]->len;
    free(ap);
    printf("base: %ld params (%.2f MB) — FROZEN\n", np_base, np_base*4.0f/1048576.0f);
    printf("LoRA: %ld params (%.1f KB) — %.2f%% of base\n",
           np_adapter, np_adapter*4.0f/1024.0f, 100.0f*np_adapter/np_base);

    nt_schedule sched = nt_schedule_cosine(base_lr, steps/10, steps, base_lr*0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    printf("\nSFT training...\n");
    printf("─────────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float first_loss = 0, best_loss = 99.0f;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        int off = rand() % (n_tokens - CTX - 1);

        nt_tape_start();
        int loss_idx = forward(m, encoded + off, encoded + off + 1);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) first_loss = lv;
        if (lv < best_loss) best_loss = lv;

        nt_tape_backward(loss_idx);
        if (!nt_nan_guard_check(&guard)) { nt_tape_clear(); continue; }
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, lv);
        nt_tape_clear();

        if ((step+1) % LOG_EVERY == 0 || step == 0) {
            printf("  step %4d/%d | train %.4f | best %.4f | lr %.2e | %.1fs\n",
                   step+1, steps, lv, best_loss, lr, (now_ms()-t0)/1000.0);
            fflush(stdout);
        }
        if ((step+1) % CKPT_EVERY == 0) {
            save_adapters(m, SFT_PREFIX);
            printf("  ──── ckpt %d saved\n", step+1);
            fflush(stdout);
        }
    }
    double total_s = (now_ms()-t0)/1000.0;

    printf("─────────────────────────────────────────────────────\n");
    printf("  train: %.4f → best %.4f\n", first_loss, best_loss);
    printf("  time:  %.0fs (%.1f min) | %.2f steps/s\n", total_s, total_s/60.0, steps/total_s);
    printf("  nans:  %d\n", guard.total_nan_count);

    printf("\n── saving LoRA adapter ──\n");
    save_adapters(m, "janus_sft_leo");
    printf("  janus_sft_leo.bin (%.1f KB, %ld params)\n", np_adapter*4.0f/1024.0f, np_adapter);
    save_adapters(m, SFT_PREFIX);

    model_free(m); free(encoded);
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  Janus SFT done. Base frozen, %ld LoRA params trained.\n", np_adapter);
    printf("════════════════════════════════════════════════════════\n");
    return 0;
}

# Microjanus weights (notorch-trained)

Three variants of the ~1.5-2.25 M parameter Janus architecture trained on
the Sonar 241KB dataset using [notorch](https://github.com/ariannamethod/notorch).
Trainer and inference code: [`../notorch-train/`](../notorch-train/).

| File | Params | Training | best train | val | Notes |
|------|--------|----------|------------|-----|-------|
| `microjanus_single_10k.bin` | 1.57M | 5000 steps + resume 5000 | **1.22** | **2.70** | Best val. Single weight per linear. |
| `microjanus_dual_sym_5k.bin` | 2.25M | 5000 steps | 1.55 | 3.32 | Dual weights, α_init = 0 → σ=0.5. α did not move from init. |
| `microjanus_dual_asym_5k.bin` | 2.25M | 5000 steps | 1.84 | 3.36 | Dual weights, α_init = 2.0 → σ=0.88, W_B × 0.5. α did not diverge. |

All runs: 0 NaN, 8GB Mac with Apple Accelerate BLAS. Tokenizer: Arianna
BPE 2048 (`../notorch-train/arianna_bpe_merges.txt`).

## Architecture (all three)

- VOCAB 2048, CTX 128, DIM 128, 4 heads × HEAD_DIM 32, 4 layers, HIDDEN 256
- Triple attention per layer: MHA (Q·K^T/√d) + RRPRAM (X·Wr) + Janus Echo
  (W^T·W on echo projection). Equal 1/3 blend.
- RoPE on Q, K. RMSNorm. SwiGLU FFN. Chuck optimizer. Cosine LR schedule.
- `dual_*` variants have `σ(α)·W_A + σ(−α)·W_B` per linear projection.

## Training takeaway

Dual weights did not outperform single-long on this 241K corpus. Two
matrices need larger data to specialize; here the gain is from implicit
ensemble of Xavier-init matrices, not from learned α-blend. Dual becomes
relevant at 20-30M parameter scale on FineWeb-class corpora.

## LoRA SFT adapter

| File | Size | Params | Training | best train |
|------|------|--------|----------|------------|
| `microjanus_sft_leo_adapter.bin` | 96 KB | 24,576 (1.04% of base) | 1500 steps, rank=8, α=16 | **4.99** |

Trained on 150KB chunk of the Leo dataset (Q/A dash-dialog, Leo voice).
Base weights frozen via `nt_tape_freeze_param()`; only rank-8 adapters
on Q/K/V projections are updated. The adapter adds a δ layer on top of
the Sonar-trained γ base — this is the θ = ε + γ + αδ decomposition
made concrete: same substrate, different voice.

Sample output (base `microjanus_dual_sym_5k.bin` + adapter):

```
[5] > loss is love. At ing, and the same runninguarglast hoking because
      biology, and ved for noth, the shifting is the crack is better
      at a knocked feoral he had that can be measured, mean who ds is
      absence le.
[7] > haze is the south than country that have always been describe
      a project should bed fe lives by the woman who rettes — riglamb
      signal is meaninglessation.
```

Leo voice leaking in: *measured, biology, project, signal, object, cost*.
Sonar core preserved: *soup, bone, knock, haze, forty minutes, loss is love*.

## Running inference

```bash
# Build from source
cd ../notorch-train/
make

# Base chain inference (proper Janus: triple attention + calendar drift
# + Schumann + AML physics + SPA)
./infer_janus_sonar_chain ../weights/microjanus_dual_sym_5k.bin "seed"

# Base + LoRA adapter (Leo voice δ over Sonar γ)
./infer_janus_sft ../weights/microjanus_dual_sym_5k.bin \
                  ../weights/microjanus_sft_leo_adapter.bin "seed"

# Train your own SFT adapter on any corpus
./train_janus_sft ../weights/microjanus_dual_sym_5k.bin your_corpus.txt 1500 1e-3
```

The chain binary performs 8-step bidirectional generation with
calendar-drift compass, Schumann temperature modulation, best-of-3
candidates per step, and SPA reseed of the weakest sentence.

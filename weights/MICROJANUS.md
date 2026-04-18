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

## Running inference

```bash
# Build from source
cd ../notorch-train/
make infer_janus_sonar_chain

# Run proper Janus chain inference on best weights
./infer_janus_sonar_chain ../weights/microjanus_single_10k.bin "seed text here"
```

The chain binary performs 8-step bidirectional generation with
calendar-drift compass, Schumann temperature modulation, best-of-3
candidates per step, and SPA reseed of the weakest sentence.

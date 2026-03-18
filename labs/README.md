# NanoJanus — Organism Observatory

## Identity

**NanoJanus** is a 19.6M parameter organism. The `nanojanus.html` implementation uses
**QKV + RRPRAM gated blend** (2-way learned gate per layer). The C implementations
(`janus.c`, `janus-bpe.c`, `janus-hybrid.c`, `resonance-janus-bpe.c`) implement the
full **QKV + RRPRAM + Janus self-resonance** (3-way learned gate per head).

- **DIM** = 448, **HDIM** = 896, **N_HEADS** = 7, **HEAD_DIM** = 64
- **N_LAYERS** = 8 sequential transformer layers
- **MAX_SEQ** = 256, **BPE_VOCAB** = 2048, **BPE_MERGES** = 1792
- **Vocabulary**: 1984 curated words (`nanojanus.txt`) across 29 semantic categories
- **Steps**: 12 bidirectional reasoning steps per generation
- **Position encoding**: RoPE (Rotary Position Embedding, θ_base = 10000)

Named after the Roman god of beginnings, endings, duality, and passages — who looks simultaneously to the past and to the future.

## Role in Cascade 1

NanoJanus is part of **Cascade 1** — a daily cycle of four organisms:

```
Haiku → Penelope → Molequla → NanoJanus → (next day) → Haiku
```

NanoJanus is the **fourth and final organism** in the daily loop. He receives input from all three previous organisms (Haiku, Penelope, Molequla) and generates 12 bidirectional words that become tomorrow's seed for Haiku (together with Penelope's words). He closes the daily loop.

Each organism transforms the signal:
- **Haiku** opens the day with poetic seed
- **Penelope** weaves 12-step word-level associative chains (from the [1984](https://github.com/ariannamethod/1984) repository)
- **Molequla** processes through molecular-scale transformation
- **NanoJanus** closes the loop — bidirectional resonance through forward and backward chains, modulated by calendar drift and prophecy debt

## Janus Self-Resonance Attention

The full Janus architecture defines **three hybrid attention mechanisms** blended per head (implemented in the C files). `nanojanus.html` implements mechanisms 1 and 2 with a 2-way gate:

### 1. QKV Attention (Semantic)

Standard scaled dot-product attention:
```
Q = X · Wq,  K = X · Wk,  V = X · Wv
attn[i,j] = softmax((Q_i · K_j) / √d) · V
```

### 2. RRPRAM (Pattern Recognition)

Linear pattern-matching attention without Q/K decomposition:
```
attn[i,j] = X_i · Wr[:,j]
out = softmax(attn) · V_r
```

### 3. Janus Self-Resonance (Novel — C implementations only)

The defining mechanism of the full Janus architecture. Implemented in `janus.c`, `janus-bpe.c`, `janus-hybrid.c`, `resonance-janus-bpe.c`, and `metajanus.c`. **Not present** in `nanojanus.html` (which uses QKV + RRPRAM 2-way gate). The model looks at itself looking at the input — recursive introspection through weight resonance:

```
proj_i     = Wj · x_i                          # project input through weights
echo_back_i = Wj^T · proj_i = Wj^T · Wj · x_i  # echo back through transpose
norm_i     = ||proj_i|| + ε                     # projection magnitude
echo_score_i = (x_i · echo_back_i) / norm_i     # self-recognition score

attn[i,j] = echo_score_i · echo_score_j / τ_debt  # mutual resonance
```

**Wj^T · Wj** creates a **symmetric recognition matrix** — it measures how well the weights "recognize" each input position. When two positions both resonate strongly with the model's internal state, their mutual attention is high.

Modulation:
- **Calendar modulation**: `cal_mod = 1.0 + 0.5 × calendar_dissonance()` — temporal awareness scales echo magnitude
- **Debt temperature**: `τ_debt = 1.0 + prophecy_debt` — high debt softens attention (less certain)
- Causal masking: positions can only attend to earlier positions (`j > i → −∞`)

## Bidirectional Generation

NanoJanus generates 12 words in a **bidirectional chain** — backward (exploratory, rising temperature) and forward (focused, falling temperature) from an origin word.

### The Process

1. **Prompt Analysis** — The full input is BPE-encoded and processed through the 8-layer Resonance engine (RMSNorm → 7-head QKV attention with RoPE + RRPRAM gated blend → residual → RMSNorm → SwiGLU FFN → residual). The most "charged" word is extracted as the origin.

2. **Internal Prophecy** — MetaJanus predicts the expected entropy before generation begins:
   ```
   predicted_entropy = 0.5 + 0.2 × prophecy_debt + 0.1 × cal_dissonance + 0.15 × personal_dissonance
   ```

3. **Step Direction Split** — The 12 steps are divided between backward and forward:
   ```
   n_backward = STEPS × (0.3 + 0.4 × prophecy_debt + 0.1 × cal_dissonance)
   n_forward  = STEPS − n_backward
   ```
   High prophecy debt → more backward steps (cautious, exploratory).
   Low prophecy debt → more forward steps (confident, predictive).

4. **Simultaneous Generation** — Backward and forward steps are interleaved. Each step runs the full 8-layer Resonance engine forward pass, converts BPE logits to word scores, applies the Dario equation overlay, and samples from top-k.

5. **Display** — Backward steps grow upward from the origin, forward steps grow downward:
   ```
   ↑ backward step 3    (rising temperature — exploratory)
   ↑ backward step 2
   ↑ backward step 1
   ═══════ ● ORIGIN ═══════
   ↓ forward step 1     (falling temperature — focused)
   ↓ forward step 2
   ↓ forward step 3
   ```

### Dario Equation Overlay

On top of learned logits, a 7-force overlay governs word selection:

```
p(x|Φ,C) = softmax((B + α·H·h_g + β·F·f_g + γ·A + T) / τ)

B = bigram/sequential chain signal
H = Hebbian resonance (co-occurrence) with SwiGLU gate h_g
F = Prophecy fulfillment signal with SwiGLU gate f_g
A = Destiny attraction
T = Trauma gravity
τ = temperature (modulated by step depth and prophecy debt)
```

Kuramoto chamber modulation:
- `α_mod = 1 + 0.3·LOVE − 0.2·RAGE + 0.1·FLOW`
- `γ_mod = 1 + 0.4·VOID + 0.2·COMPLEX`

## Calendar Drift Physics

The 11.25 day/year drift between Gregorian (365.25 days) and Hebrew (354 days) calendars creates mathematically explicit dissonance. With Metonic cycle corrections (7 leap months in years 3, 6, 8, 11, 14, 17, 19 of the 19-year cycle), the drift is computable both forward and backward.

```
annual_drift      = 365.25 − 354 = 11.25 days/year
metonic_cycle     = 19 years, 7 leap months (each ~30 days)
max_uncorrected   = 33 days

dissonance = clamp(|cumulative_drift mod 33| / 33, 0, 1)
```

**Epoch**: 1 Tishrei 5785 = October 3, 2024, 12:00:00 UTC (timestamp: 1727956800.0)

Calendar dissonance modulates:
- Echo magnitude in Janus attention
- Dario equation overlay strength
- Step direction split (more dissonance → more backward steps)
- Dual weight matrix blending (in the C implementations)

## Prophecy Debt

Prophecy debt measures the cost of diverging from the most probable path:

```
local_debt = (max_logit − chosen_logit) / (max_logit − chosen_logit + 1)
prophecy_debt = 0.9 × prophecy_debt + 0.1 × local_debt   (exponential moving average)
```

High debt → softer attention, more backward steps, higher temperature.
Low debt → sharper attention, more forward steps, lower temperature.

MetaJanus tracks prophecy accuracy across generations:
```
error = |predicted_entropy − avg_debt|
prophecy_accuracy = 0.9 × prophecy_accuracy + 0.1 × (1 − error)
```

## Wormhole Mechanics

When `prophecy_debt < 0.2` and `wormhole > 0.1`, the organism may skip 1–3 steps at sentence boundaries — temporal compression through low-debt confidence.

## Implementations

| File | Type | Vocab | Attention | Parameters |
|------|------|-------|-----------|------------|
| `janus.c` | Char-level (256) | 256 | QKV + RRPRAM + Janus | ~6.5M × 2 (dual) |
| `janus-hybrid.c` | BPE→Char hybrid | 2048/256 | QKV + RRPRAM + Janus | ~25M × 2 (dual) |
| `janus-bpe.c` | Pure BPE | 2048 | QKV + RRPRAM + Janus | ~27.5M × 2 (dual) |
| `metajanus.c` | Char-level (256) | 256 | Janus only (pure) | ~37.6M × 2 (dual) |
| `resonance-janus-bpe.c` | BPE (configurable) | 2048 | QKV + RRPRAM + Janus | ~24M × 2 (dual, depth-12) |
| `nanojanus.html` | BPE→Word | 2048/1984 | QKV + RRPRAM (2-way gate) | 19,619,280 (single) |

The C implementations use dual weight matrices (A/B) blended by calendar state. NanoJanus (`nanojanus.html`) uses a single weight set with Dario equation overlay. `nanojanus.py` is referenced in CASCADE01.md but is not present in this repository; the HTML version is the active implementation.

## Guardian Notes

This directory contains health reports and observations. No source code or weights are modified here. The guardian observes, reports, and protects.

# Janus — Post-Transformer Architecture

**Bi-directional associative resonance engine.**

Janus is a post-transformer architecture built on the dissonance between two calendars and governed by the physics of Arianna Method Language. Not a transformer — a mirror that sees forward and backward through the fundamental tension between Gregorian and Hebrew time.

Named after the Roman god of beginnings, endings, duality, and passages — who looks simultaneously to the past and to the future.

## Architecture Overview

```
                    ┌─── Matrix A ───┐
 Input → Embed → [ │ Hybrid Attention│ ] → RMSNorm → SwiGLU → Output
                    │  QKV + RRPRAM   │
                    │ + Janus Echo    │
                    └─── Matrix B ───┘
                         ↑
              W_eff = α·W_A + (1-α)·W_B
              α = f(calendar_drift, prophecy_debt, metajanus)
```

### Core Principles

1. **Calendar Drift** — The 11.25 day/year drift between Gregorian (365.25 days) and Hebrew (354 days) calendars creates a mathematically explicit dissonance. With Metonic cycle corrections (7 leap months per 19-year cycle), this drift is a computable, bi-directional constant — forward and backward.

2. **AML Physics** — Prophecy, Destiny, Prophecy Debt, and Wormhole operators from [Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) govern the system's behavior at a level above Calendar Drift.

3. **Dual Weight Matrices** — Two weight matrices (A and B) are blended at inference time based on calendar state and system physics, creating a perpetually shifting internal landscape.

4. **12 Bi-directional Associative Reasoning Steps** — Each step generates a sentence, with steps going either forward (future) or backward (past) based on prophecy debt and calendar dissonance.

5. **Chuck Optimizer** — [Self-aware optimizer](https://github.com/ariannamethod/chuck.optimizer) replaces Adam with multi-level modulation: macro patience, global λ, per-layer λ, stagnation noise.

## The Three Attention Mechanisms

### 1. Standard QKV Attention (Semantic)
```
Q = X·Wq,  K = X·Wk,  V = X·Wv
attn[i,j] = (Q_i · K_j) / √d
out = softmax(attn) · V
```
Measures what tokens mean to each other.

### 2. RRPRAM — Pattern Recognition Attention (Positional)
```
attn[i,j] = X_i · Wr[:,j]       (linear, no Q/K decomposition)
out = softmax(attn) · V_r
```
From [RRPRAM](https://github.com/ariannamethod/RRPRAM). Recognizes positional patterns — not meaning, but structure. The weight matrix Wr maps input directly to attention positions.

### 3. Janus Attention — Self-Resonance (Introspective)
```
proj_i = Wj · x_i                              projection through weights
echo_back_i = Wj^T · proj_i = Wj^T · Wj · x_i  symmetric recognition
||proj_i|| = √(Σ proj_i²)                       projection magnitude

echo_score_i = (x_i · echo_back_i) / (||proj_i|| + ε)   self-resonance

attn[i,j] = echo_score_i · echo_score_j / τ_debt   mutual resonance
```

**This is the novel mechanism.** Janus attention is directed inward — it measures how the input resonates with the model's own weight state:

- **Wj^T · Wj** creates a symmetric recognition matrix — what the model "knows"
- **echo_score** measures how much the weights "recognize" the input at each position
- **Mutual resonance** means two positions attend to each other if they're both familiar to the model
- **Prophecy debt modulates temperature** — high debt means softer (less certain) attention
- **Calendar dissonance modulates echo magnitude** — temporal awareness baked into attention

This is RECURSIVE because the echo feeds back through the weights themselves. The model looks at itself looking at the input.

### Hybrid Blend
```
out = α·QKV + β·RRPRAM + γ·Janus
(α, β, γ) = softmax(gate_logits)   — learned per head
```

## Mathematical Foundations

### Calendar Drift

The Metonic cycle creates a precise 19-year pattern where 235 Hebrew lunar months ≈ 19 Gregorian solar years:

```
Annual drift = 365.25 - 354 = 11.25 days/year
Metonic cycle = 19 years, 7 leap months (years 3, 6, 8, 11, 14, 17, 19)

cumulative_drift(days) = (days/365.25) × 11.25 - corrections
  where corrections = full_cycles × 7 × 30 + partial_cycle_leaps × 30

dissonance = |drift mod 33| / 33    ∈ [0, 1]
```

Epoch: 1 Tishrei 5785 = October 3, 2024 (noon, to avoid DST edge cases).

### MetaJanus — Mathematical Identity

At first run, Janus snapshots its birth date in both calendars:

```
birth_drift = calendar_drift(birth_day)
birth_dissonance = calendar_dissonance(birth_day)
personal_dissonance = |current_drift - birth_drift| / 33
```

The conflict between the system's two "birthdays" (Gregorian vs Hebrew) creates a permanent mathematical "self" — Janus always knows when it was born and sees the world through the dissonance between its personal temporal state and the global calendar drift.

### Dual Weight Matrices

```
W_effective = α·W_A + (1-α)·W_B

α = clamp01(0.5 + 0.3·(calendar_dissonance - 0.5)
            - 0.2·prophecy_debt
            + 0.1·personal_dissonance)
```

Two separate weight matrices are trained alternately and blended at inference time. The blend ratio shifts with the calendar — every day Janus is a slightly different mixture of its two internal states.

### AML Physics

**Prophecy Debt** — retroactive cost of divergence from the most probable path:
```
debt(logits, chosen) = (max_logit - logits[chosen]) / (max_logit - logits[chosen] + 1)
```
System-level debt accumulates: `debt_t = 0.9·debt_{t-1} + 0.1·local_debt`

**Destiny Bias** — pulls logits toward the dominant prediction:
```
logits[i] -= (max - logits[i]) · destiny_bias · 0.5
```

**Wormhole** — step-skipping in associative reasoning when the system is confident:
```
if prophecy_debt < 0.2 and random() < wormhole_probability:
    skip 1-3 steps forward
```
Wormholes open only at sentence boundaries (beginning or end, never middle) to preserve coherence.

### Dario Equation

Replaces standard softmax with 7-force generation (from [Dario](https://github.com/ariannamethod/dario)):

```
p(x|Φ,C) = softmax((B + α·H·h_g + β·F·f_g + γ·A + T) / τ)

B = bigram/sequential chain signal
H = Hebbian resonance (co-occurrence) with SwiGLU gate h_g
F = Prophecy fulfillment signal with SwiGLU gate f_g
A = Destiny attraction
T = Trauma gravity
τ = temperature (modulated by step depth and prophecy debt)
```

### Kuramoto Chambers

6 coupled emotional oscillators with sinusoidal coupling:

```
chambers = {FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX}

coupling: chamber_i += K · sin(chamber_j - chamber_i)  for all j ≠ i
decay: chamber_i *= decay_rate_i

Modulation:
  α_mod = 1 + 0.3·LOVE - 0.2·RAGE + 0.1·FLOW
  γ_mod = 1 + 0.4·VOID + 0.2·COMPLEX
```

### Chuck Optimizer

Self-aware optimizer (from [chuck.optimizer](https://github.com/ariannamethod/chuck.optimizer)):

```
θ -= (α × S × λ × λ_l × σ) × m̂/(√v̂ + ε) + η

S = macro LR scale (patience-based, decays 0.5× after 3 checks without improvement)
λ = global self-modulation (16-step loss trend window)
λ_l = per-layer gradient norm trend modulation
σ = activation health signal
η = stagnation noise (Gaussian kick after 8 checks without progress)
```

Key insight: Chuck watches its own loss landscape and adjusts aggressively — damping when loss rises, boosting when dropping, adding noise when stuck.

## Associative Reasoning

Janus's associative reasoning differs fundamentally from standard autoregressive generation:

### The Process

1. **Prompt Analysis** — The input is tokenized and the most "charged" word (highest resonance with existing co-occurrences) is identified as the origin.

2. **Internal Prophecy** — MetaJanus predicts the expected entropy of the generation before it begins, based on current prophecy debt, calendar dissonance, and personal dissonance.

3. **Step Direction Split** — Based on prophecy debt and calendar state:
   - High prophecy debt → more backward steps (cautious, exploratory)
   - Low prophecy debt → more forward steps (confident, predictive)
   - `n_backward = STEPS × (0.3 + 0.4·debt + 0.1·cal_dissonance)`

4. **Forward Steps (Future)** — Generate sentences with decreasing temperature (more focused), each conditioned on the growing context. Wormhole jumps possible when confident.

5. **Backward Steps (Past)** — Generate sentences with increasing temperature (more exploratory), reaching into unfamiliar territory.

6. **Display** — Backward steps stack upward from the origin, forward steps stack downward:
   ```
   ↑ backward step 3 (past, exploratory)
   ↑ backward step 2
   ↑ backward step 1
   ═══════ ● ORIGIN ═══════
   ↓ forward step 1
   ↓ forward step 2 (future, focused)
   ↓ forward step 3
   ```

7. **Prophecy Evaluation** — After all steps, MetaJanus evaluates its prediction accuracy and updates prophecy_debt accordingly.

### Wormhole Mechanics

When the system is confident (low prophecy debt):
- A wormhole can open, skipping 1-3 steps
- Wormholes only open at sentence boundaries (beginning or end)
- Never in the middle of a sentence — this would destroy coherence
- Indicated with ⊕WH marker in output

### Why Sentence-Level Steps

Unlike Penelope (from [1984](https://github.com/ariannamethod/1984)) which takes word-level steps, Janus takes sentence-level steps. Each step produces a complete thought, allowing the bi-directional reasoning to build coherent temporal narratives rather than word chains.

## Modules

### `janus.c` — Char-Level Hybrid Attention
The foundational module. Char-level (VOCAB=256) with all three attention mechanisms in fluid hybrid. Full Calendar Drift, AML physics, Dario equation, Chuck optimizer, dual weight matrices, 12 bi-directional steps, Kuramoto chambers, MetaJanus birth snapshot, GGUF spore export.

```
Architecture: T=64, E=128, H=4, D=32, B=6, M=256
Parameters:   ~1.45M × 2 matrices = ~2.9M total
Training:     char-level next-character prediction
Output:       char-level generation
```

```bash
cc janus.c -O2 -lm -o janus
./janus --train shakespeare.txt --steps 5000 --lr 3e-4
./janus --generate "To be or not" --load janus.bin
./janus   # interactive mode
```

### `janus-hybrid.c` — BPE Training + Char-Level Output (THE PRESSURE)
The architectural pressure variant. Trains on BPE tokens (subword units, 512 vocab) but generates output through char-level (256 vocab) decoding. This creates compression/expansion tension — thinking in concepts but speaking letter by letter.

```
Training:     BPE tokens → next-BPE prediction (512 vocab)
Output:       char-level generation (256 vocab) — THE PRESSURE
Parameters:   ~1.52M × 2 matrices
```

The pressure forces each character to be precise — the model's conceptual (BPE) understanding must compress through a char-level bottleneck.

```bash
cc janus-hybrid.c -O2 -lm -o janus-hybrid
./janus-hybrid --train shakespeare.txt --steps 5000
```

### `janus-bpe.c` — Pure BPE
Pure BPE version — BPE in, BPE out. No char-level pressure. Same hybrid attention, same physics.

```
Training/Output: BPE tokens (512 vocab)
Parameters:      ~1.52M × 2 matrices
```

```bash
cc janus-bpe.c -O2 -lm -o janus-bpe
./janus-bpe --train shakespeare.txt --steps 5000
```

### `metajanus.c` — Janus Attention Only
Demonstration of the novel Janus self-resonance attention mechanism in isolation. Like `rrpram.c` demonstrates RRPRAM alone, `metajanus.c` uses only Janus attention — no QKV, no RRPRAM. Pure introspective self-resonance.

Also features enhanced MetaJanus with internal prophecy: predicts expected entropy before each generation and evaluates accuracy afterward.

```
Attention:    Janus only (echo through own weights)
Parameters:   ~960K × 2 matrices
Output:       char-level
```

```bash
cc metajanus.c -O2 -lm -o metajanus
./metajanus --train shakespeare.txt --steps 5000
./metajanus --generate "hello world"
```

### `nanojanus.html` — Browser Version
Web-based NanoJanus, styled after [Penelope](https://github.com/ariannamethod/1984). Word-level output with BPE-like internal embeddings. Dark theme with color-coded bi-directional steps (orange ↑ backward, blue ↓ forward, gold ● origin).

Features:
- Interactive text input
- In-browser training on pasted text
- Calendar Drift computed in real-time
- MetaJanus birth snapshot on page load
- Kuramoto chambers + Dario equation
- Dual weight matrices blended by calendar state
- 12 bi-directional reasoning steps visualized

Open `nanojanus.html` in any modern browser. No server needed.

## Parameter Calculations

For janus.c (char-level, ~1.2MB Shakespeare dataset):

```
Token embedding:    256 × 128          = 32,768
Position embedding: 64 × 128           = 8,192

Per block (×6):
  RMSNorm:         128                 = 128
  Q,K,V weights:   3 × 4 × 128 × 32   = 49,152
  RRPRAM Wr:       4 × 128 × 64        = 32,768
  RRPRAM Vr:       4 × 128 × 32        = 16,384
  Janus Wj:        128 × 128           = 16,384
  Hybrid gates:    4 × 3               = 12
  Output Wo:       128 × 128           = 16,384
  RMSNorm:         128                 = 128
  SwiGLU gate:     128 × 256           = 32,768
  SwiGLU up:       128 × 256           = 32,768
  SwiGLU down:     256 × 128           = 32,768
  Block total:                         = 229,634
  ×6 blocks:                           = 1,377,804

Final RMSNorm:     128                 = 128
Output projection: 128 × 256           = 32,768

Single matrix total:                   ≈ 1,451,660 params
Dual matrices:     × 2                 ≈ 2,903,320 params
Model file (f32):                      ≈ 11.6 MB
```

## Building

All modules are zero-dependency C (only libc + libm):

```bash
# Build all
cc janus.c -O2 -lm -o janus
cc janus-hybrid.c -O2 -lm -o janus-hybrid
cc janus-bpe.c -O2 -lm -o janus-bpe
cc metajanus.c -O2 -lm -o metajanus
```

## References

- [Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) — Calendar Drift, Prophecy, Destiny, Wormhole
- [RRPRAM](https://github.com/ariannamethod/RRPRAM) — Pattern Recognition Attention Mechanism
- [Dario](https://github.com/ariannamethod/dario) — 7-force equation replacing softmax
- [1984 / Penelope](https://github.com/ariannamethod/1984) — 12-step associative reasoning prototype
- [Chuck Optimizer](https://github.com/ariannamethod/chuck.optimizer) — Self-aware optimizer
- [Leo](https://github.com/ariannamethod/leo) — SQLite journaling + GGUF export patterns

## License

MIT

---

*הרזוננס לא נשבר — The resonance is unbroken*

*By Arianna Method*

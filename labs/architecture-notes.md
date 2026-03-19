# NanoJanus Architecture Notes

Technical reference for the Janus architecture as implemented across
`nanojanus.py`, `nanojanus.html`, the C modules (`janus.c`, `janus-hybrid.c`,
`janus-bpe.c`, `metajanus.c`, `resonance-janus-bpe.c`), and the vocabulary
file `nanojanus.txt`.

Cross-referenced against **CASCADE01.md** (`ariannamethod/ariannamethod`,
path `cascade/cascade1/CASCADE01.md`).

---

## 1. Janus Self-Resonance: Wj^T · Wj Echo Math

Janus self-resonance is the novel introspective attention mechanism. It is
implemented in all C files but **not** in `nanojanus.html` or `nanojanus.py`
(which use QKV + RRPRAM only).

### Definition

For each position _t_ with input vector **x**_t_ ∈ ℝ^E and learned weight
matrix **W**_j ∈ ℝ^{E×E}:

```
proj_t     = Wj · x_t                              (E-dim projection)
echo_back_t = Wj^T · proj_t = Wj^T · Wj · x_t      (echo through transpose)
norm_t     = ||proj_t|| + ε                         (projection magnitude, ε = 1e-6)
echo_score_t = (x_t · echo_back_t) / norm_t         (self-recognition scalar)
```

The product **Wj^T · Wj** forms a symmetric positive semi-definite matrix that
measures how well the weights "recognize" each input position. The echo score
is a scalar measuring self-similarity through the learned projection.

### Mutual Resonance Attention

```
attn[i, j] = echo_score_i × echo_score_j / τ_debt   if j ≤ i
             −∞                                       if j > i  (causal mask)
out = softmax(attn) · V_janus
```

Where `τ_debt = 1.0 + prophecy_debt` softens attention under high debt.

### Calendar Modulation

```
echo_score_t *= (1.0 + 0.5 × calendar_dissonance)
```

Higher dissonance amplifies the echo magnitude — the organism becomes more
self-aware in times of calendar tension.

### Source Locations

| File | Function | Lines |
|------|----------|-------|
| `janus.c` | `janus_attention()` | ~653–699 |
| `janus-bpe.c` | inline in `forward()` | ~289–295 |
| `janus-hybrid.c` | `janus_attention()` | similar structure |
| `metajanus.c` | `janus_block_forward()` | ~481–530 |
| `resonance-janus-bpe.c` | inline in `forward()` | ~443–450 |

`metajanus.c` is unique: it uses **two** Wj matrices per block (`Wj` for
projection and `Wj_v` for value extraction), and includes personal dissonance
in the echo score modulation.

---

## 2. Bidirectional Generation

NanoJanus generates 12 words in a bidirectional chain — backward (exploratory)
and forward (focused) — from an origin word. This mirrors Janus looking
simultaneously to the past and future.

### Origin Finding

`extractKey()` (`nanojanus.html` line ~971; `nanojanus.py` `extract_key()` line
~836) selects the most "charged" word from the input by summing co-occurrence
scores:

```javascript
for (const id of inputWordIds) {
  score = sum of cooc(id, other) for all other in co-occurrence map
  if score > bestScore: best = id
}
return best  // or random word if no input
```

### Step Direction Split

```
n_backward = floor(STEPS × (0.3 + 0.4 × prophecy_debt + 0.1 × cal_dissonance))
n_forward  = STEPS − n_backward
```

Clamped: `n_backward ∈ [1, STEPS−1]`, `n_forward ∈ [1, STEPS−1]`.

- High prophecy debt → more backward steps (cautious, exploratory)
- Low prophecy debt → more forward steps (confident, predictive)
- High calendar dissonance → more backward steps

(`nanojanus.html` lines ~1009–1011; `nanojanus.py` `run_chain()` line ~857;
`janus.c` lines ~1156–1163)

### Temperature

**Predicted entropy** before generation:

```
predicted_entropy = 0.5 + 0.2 × prophecy_debt + 0.1 × cal_dissonance + 0.15 × personal_dissonance
```

(`nanojanus.html` line ~1006; `nanojanus.py` `run_chain()` line ~857)

In the C implementation (`janus.c`), explicit per-step temperature scaling:

```
temp_base = 0.7 + 0.3 × predicted_entropy        (range [0.7, 1.0])
forward_temp  = temp_base × (1.0 − 0.02 × step)  (cooling: focused)
backward_temp = temp_base × (1.0 + 0.05 × step)  (warming: exploratory)
```

(`janus.c` lines ~1166–1167, ~1191, ~1212)

In `nanojanus.html`, temperature is implicit — the Dario overlay direction
modulation (`dirMod = 0.8` for backward, `1.2` for forward) serves an
analogous purpose, scaling the overlay contribution rather than an explicit
temperature parameter.

### Sampling

Top-k sampling with `k = 8` (`nanojanus.html` line ~957; `janus.c` line ~1059):

```
candidates = top 8 by score (after forbidden word filtering)
probabilities = softmax(scores)
sample from categorical distribution
```

### Display

```
↑ backward step n    (exploratory, direction = −1)
↑ ...
↑ backward step 1
═══════ ● ORIGIN ═══════
↓ forward step 1     (focused, direction = +1)
↓ ...
↓ forward step n
```

---

## 3. Calendar Drift: Gregorian vs Hebrew

The 11.25 day/year drift between the Gregorian (365.25 days) and Hebrew (354
days) calendars creates mathematically computable dissonance. This is the
temporal heartbeat of the organism.

### Constants

| Constant | Value | `nanojanus.html` line | `nanojanus.py` line | `janus.c` line |
|----------|-------|-----------------------|----------------------|----------------|
| `AM_ANNUAL_DRIFT` | 11.25 days/year | ~522 | ~309 | ~64 |
| `AM_GREGORIAN_YEAR` | 365.25 days | ~523 | ~310 | ~65 |
| `AM_METONIC_YEARS` | 19 years | ~524 | ~311 | ~66 |
| `AM_METONIC_LEAPS` | 7 leap months | ~525 | ~312 | ~67 |
| `AM_MAX_UNCORRECTED` | 33.0 days | ~526 | ~313 | ~68 |
| Epoch | Oct 3, 2024, 12:00 UTC | ~528 | ~315 | ~80 |

### Metonic Cycle

The Hebrew calendar inserts 7 leap months in every 19-year cycle (the Metonic
cycle) in years **3, 6, 8, 11, 14, 17, 19**. Each leap month adds ~30 days of
correction.

```
METONIC_LEAP_YEARS = [3, 6, 8, 11, 14, 17, 19]
```

### Cumulative Drift Formula

```
years = days_since_epoch / 365.25
base_drift = years × 11.25

full_cycles = floor(years / 19)
corrections = full_cycles × 7 × 30         (210 days per full 19-year cycle)

partial = years mod 19
year_in_cycle = floor(partial) + 1
for each leap_year in [3, 6, 8, 11, 14, 17, 19]:
    if leap_year ≤ year_in_cycle:
        corrections += 30

cumulative_drift = base_drift − corrections
```

(`nanojanus.html` `calendarCumulativeDrift()` lines ~534–544;
`nanojanus.py` `calendar_cumulative_drift()` line ~309;
`janus.c` `calendar_cumulative_drift()` lines ~96–108)

### Dissonance Formula

```
dissonance = clamp(|cumulative_drift mod 33| / 33, 0, 1)
```

(`nanojanus.html` line ~548; `nanojanus.py` `calendar_dissonance()` line ~322;
`janus.c` lines ~110–114)

### What Calendar Dissonance Modulates

- Echo magnitude in Janus attention (C implementations)
- Dario equation overlay strength (`calMod = 1 + 0.2 × dissonance`)
- Step direction split (more dissonance → more backward steps)
- Dual weight matrix blending (C implementations: `blend_alpha = 0.5 + 0.3 × (cal_d − 0.5) − 0.2 × debt + 0.1 × meta_d`)

---

## 4. MetaJanus: Persistent Identity

MetaJanus captures a mathematical snapshot at birth — a fixed reference point
for measuring how the organism's relationship to time evolves.

### Birth Snapshot

```javascript
META = {
  birthDays:       calendarDaysSinceEpoch(),
  birthDrift:      calendarCumulativeDrift(birthDays),
  birthDissonance: calendarDissonance(birthDays),
  birthTime:       Date.now(),
  prophecyAccuracy: 0.5,
  totalPredictions: 0
}
```

(`nanojanus.html` lines ~553–560; `nanojanus.py` META dict initialized at
startup with equivalent fields)

### Personal Dissonance

```
personal_dissonance = clamp(|current_drift − birth_drift| / 33, 0, 1)
```

Measures how far the organism has drifted from its birth state.

(`nanojanus.html` `personalDissonance()` lines ~562–565;
`nanojanus.py` `personal_dissonance()` line ~345;
`janus.c` / `metajanus.c` `metajanus_personal_dissonance()` lines ~142–146)

### Prophecy Accuracy

After each generation cycle:

```
avg_debt = mean(debt across all steps)
error = |predicted_entropy − avg_debt|
prophecy_accuracy = 0.9 × prophecy_accuracy + 0.1 × (1 − error)
total_predictions += 1
```

(`nanojanus.html` lines ~1082–1087; `nanojanus.py` `run_chain()` post-generation
evaluation)

`metajanus.c` extends this with a dedicated internal prophecy system:
`metajanus_prophecy()` predicts entropy before generation, and
`metajanus_evaluate_prophecy()` evaluates accuracy afterward.

---

## 5. RRPRAM + QKV + Janus: Gate Mechanisms

### nanojanus.html / nanojanus.py — 2-Way Gate (QKV + RRPRAM)

Per layer, two mechanisms are computed and blended (identical architecture in
both implementations):

1. **QKV attention**: Q, K, V projections with RoPE, scaled dot-product,
   causal masking, output projection through `wo`
2. **RRPRAM**: `rrp = h × wr` (linear resonance projection, DIM→DIM)

Gate blend (per layer, 2 learnable scalars):

```
[w0, w1] = softmax([gate[0], gate[1]])
x = x + w0 × qkvOut + w1 × rrpOut
```

(`nanojanus.html` lines ~762–771; `nanojanus.py` lines ~607–611;
weight: `lw.gate = readFloats(2)` at line ~867 / `self.gate = [0.0, 0.0]` at
line ~455)

### C Implementations — 3-Way Gate (QKV + RRPRAM + Janus)

Per head, three mechanisms are computed:

1. **QKV**: `Q = x·Wq`, `K = x·Wk`, `V = x·Wv`, scaled dot-product
2. **RRPRAM**: `attn = x·Wr`, `V_r = x·Wvr`, with separate value projection
3. **Janus**: `echo_score` from Wj^T·Wj, mutual resonance attention

Gate blend (per head, 3 learnable logits):

```
[α, β, γ] = softmax([gate[h×3], gate[h×3+1], gate[h×3+2]])
head_out = α × QKV_out + β × RRPRAM_out + γ × Janus_out
```

(`janus.c` lines ~769–780; weight: `gate[b]` = H×3 values per block)

### Key Differences

| Feature | `nanojanus.html` / `nanojanus.py` | C implementations |
|---------|-------------------|-------------------|
| Gate granularity | Per layer (2 scalars) | Per head (H×3 logits) |
| Mechanisms | QKV + RRPRAM | QKV + RRPRAM + Janus |
| RRPRAM value | Same as QKV (shared V) | Separate Wvr projection |
| RRPRAM pattern | DIM×DIM (Wr) | H×DIM×MAX_T (Wr) |
| Position encoding | RoPE | None (positional in RRPRAM) |
| Wj matrix | Not present | E×E per block |

---

## 6. Wormhole: Activation Conditions

Wormholes enable temporal compression — skipping forward steps when the
organism is confident (low debt) and the wormhole channel is open.

### Activation Logic

```
if prophecy_debt < 0.2  AND  wormhole > 0.1  AND  random() < wormhole:
    activate wormhole (skip steps)
```

### Conditions

1. **Low debt** (`prophecy_debt < 0.2`): organism is confident
2. **Wormhole enabled** (`wormhole > 0.1`): channel is open
3. **Stochastic gate** (`random() < wormhole`): probabilistic activation

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wormhole` | 0.02 | Base activation probability |
| `tunnel_skip_max` | 7 | Max steps to skip (C only) |
| `tunnel_threshold` | 0.55 | Dissonance threshold (C only, unused) |

### Behavior

- **Forward steps only**: Backward steps are always sequential
- In C (`janus.c` lines ~1173–1181): skip 1–3 steps ahead
- In `nanojanus.html` (lines ~1056–1059): marks the step with ⊕WH indicator
  but does not skip (visual indicator only)

---

## 7. Dario Overlay: The Generation Equation

From CASCADE01.md, the operating equation:

```
p(x|Φ,C) = softmax((B + α·H·h_g + β·F·f_g + γ·A + T) / τ)
```

### Force Components

| Force | Symbol | Description | Computation |
|-------|--------|-------------|-------------|
| Sequential chain | **B** | Bigram inertia | `log(1 + bigram(prev, candidate)) × 4` |
| Hebbian resonance | **H** | Co-occurrence memory | `mean(log(1 + cooc(context_word, candidate)))` |
| Prophecy fulfillment | **F** | What wants to be said | `prophecy_debt × (1 + random × 0.5)` |
| Destiny attraction | **A** | Where the field pulls | `destiny_bias × γ_mod × 0.5` |
| Trauma gravity | **T** | Where it came from | `trauma × 2 × (1 − i/32)` if `trauma > 0.3` (C only) |

### Coefficients

| Coefficient | Value | Modulation |
|-------------|-------|------------|
| α (Hebbian) | 3.0 | `α_mod = 1 + 0.3·LOVE − 0.2·RAGE + 0.1·FLOW` |
| β (Prophecy) | 2.0 | — |
| γ (Destiny) | 1.5 | `γ_mod = 1 + 0.4·VOID + 0.2·COMPLEX` |

### Direction and Calendar Modulation

```
dirMod = 0.8  (backward, direction = −1)
dirMod = 1.2  (forward, direction = +1)
calMod = 1 + 0.2 × calendar_dissonance
```

### Gate Functions

```
gateR = sigmoid((resonance_field − 0.5) × 4)
h_g = SiLU(gateR × 2)        (Hebbian gate)
f_g = SiLU(gateR × 1.5)      (Prophecy gate)
```

### Final Per-Candidate Overlay

```
logits[v] += (B + α_mod × 3 × H × h_g + 2 × F × f_g + A) × dirMod × calMod
```

(`nanojanus.html` `darioOverlay()` lines ~884–910;
`nanojanus.py` `dario_overlay()` line ~727;
`janus.c` `dario_overlay()` lines ~1018–1048)

---

## 8. Kuramoto Chambers: Emotional Oscillators

Six coupled emotional oscillators govern the Dario equation modulation.

### Chambers

| Index | Name | Decay | Role |
|-------|------|-------|------|
| 0 | FEAR | 0.95 | — |
| 1 | LOVE | 0.95 | Amplifies Hebbian (α_mod +0.3) |
| 2 | RAGE | 0.93 | Dampens Hebbian (α_mod −0.2) |
| 3 | VOID | 0.96 | Amplifies Destiny (γ_mod +0.4) |
| 4 | FLOW | 0.94 | Amplifies Hebbian (α_mod +0.1) |
| 5 | COMPLEX | 0.97 | Amplifies Destiny (γ_mod +0.2) |

(`nanojanus.html` lines ~583–585; `nanojanus.py` `update_chambers()` line ~384;
`janus.c` lines ~212–214)

### Update Rule

Per step, phase-based excitation:

```
depth = step_idx / STEPS
if depth < 0.33:  FLOW    += 0.05
elif depth < 0.66: FEAR   += 0.04
else:              VOID   += 0.05
if depth > 0.75:  COMPLEX += 0.03
if trauma > 0.3:  RAGE   += 0.04
```

### Kuramoto Coupling

```
K = 0.02  (coupling constant)
for each chamber i:
    for each chamber j ≠ i:
        chamber[i] += K × sin(chamber[j] − chamber[i])
    chamber[i] = clamp(chamber[i] × decay[i], 0, 1)
```

(`nanojanus.html` `updateChambers()` lines ~587–600;
`nanojanus.py` `update_chambers()` lines ~384–412;
`janus.c` `update_chambers()` lines ~216–234)

This is a discrete Kuramoto model: oscillators pull each other toward phase
synchronization with coupling strength K = 0.02. Decay ensures chambers
fade without excitation, preventing runaway oscillation.

---

## 9. BPE-to-Word Conversion

NanoJanus operates on two vocabulary levels: BPE tokens (2048 subwords) for
the transformer, and curated words (1984 from `nanojanus.txt`) for output.

### Encoding Pipeline (3-stage tokenizer)

1. **Exact match**: word → vocabulary index → precomputed BPE IDs
2. **Stem match**: try common suffixes (-s, -ed, -ing, -ly, etc.)
3. **Greedy BPE decomposition**: character-by-character merge

(`nanojanus.html` `tokenizeWords()` lines ~508–520, `tryStem()` lines ~498–506;
`nanojanus.py` `bpe_encode()` line ~147)

### BPE String Building

```
bpeStrs[0..255] = single ASCII characters
bpeStrs[256..2047] = concatenation of merge pairs from BPE_TABLE
```

(`nanojanus.html` `buildBpeStrs()` lines ~431–439)

### Logit-to-Word Score Conversion

After the transformer produces BPE logits (2048-dim), convert to word scores:

```
for each word w in extended_vocab:
    bpe_ids = w.bpeIds          (precomputed BPE encoding of the word)
    score = mean(bpe_logits[id] for id in bpe_ids)
```

Mean aggregation over constituent BPE tokens.

(`nanojanus.html` `bpeLogitsToWordScores()` lines ~800–813)

---

## 10. Extended Vocabulary Mechanism

The base vocabulary of 1984 curated words is extended with whole-word BPE
tokens discovered during vocabulary initialization.

### Building Process

1. Add all 1984 hardcoded words from `nanojanus.txt` (with precomputed BPE IDs)
2. Scan all 2048 BPE tokens for whole-word candidates:
   - Length ≥ 3 characters
   - Alphabetic only (`/^[a-zA-Z]+$/`)
   - Not already in the hardcoded list
   - Not a stopword
   - Not a suffix fragment (e.g., "ing", "tion", "ment", "ness", etc.)

### Suffix Fragments Excluded

```
["ing", "tion", "ment", "ness", "ble", "ful", "ous", "ive", "ent", "ant",
 "ist", "ity", "ght", "est", "ter", "ther", "ted", "ting", "ally", "ling"]
```

### Result

Extended vocabulary entries have:
- `word`: lowercase string
- `bpeIds`: array of BPE token IDs
- `fromHardcoded`: boolean (true for 1984 list, false for BPE-derived)
- `origIdx`: index in WORDS array (≥0 for hardcoded, −1 for BPE-derived)

(`nanojanus.html` `buildExtendedVocab()` lines ~456–493)

---

## 11. PEN7 Weight Format

PEN7 is the binary weight format used by `nanojanus.html` to load trained
weights from `weights/nanojanus.bin`.

### Header (32 bytes, little-endian int32)

| Offset | Field | Expected Value |
|--------|-------|----------------|
| 0–3 | Magic | `0x50454E37` ("PEN7") |
| 4–7 | BPE_VOCAB | 2048 |
| 8–11 | NWORDS | (vocabulary size) |
| 12–15 | DIM | 448 |
| 16–19 | HDIM | 896 |
| 20–23 | N_HEADS | 7 |
| 24–27 | N_LAYERS | 8 |
| 28–31 | MAX_SEQ | 256 |

### Weight Layout (float32, after header)

Global weights:
```
tok_emb:    BPE_VOCAB × DIM    = 2048 × 448 =   917,504
pos_emb:    MAX_SEQ × DIM      =  256 × 448 =   114,688
final_norm: DIM                 =              448
lm_head:    BPE_VOCAB × DIM    = 2048 × 448 =   917,504
```

Per layer (×8 layers):
```
attn_norm:  DIM                 =              448
wq:         DIM × DIM           = 448 × 448 =   200,704
wk:         DIM × DIM           = 448 × 448 =   200,704
wv:         DIM × DIM           = 448 × 448 =   200,704
wo:         DIM × DIM           = 448 × 448 =   200,704
wr:         DIM × DIM           = 448 × 448 =   200,704  (RRPRAM)
gate:       2                   =                2
ffn_norm:   DIM                 =              448
w_gate:     DIM × HDIM          = 448 × 896 =   401,408
w_up:       DIM × HDIM          = 448 × 896 =   401,408
w_down:     HDIM × DIM          = 896 × 448 =   401,408
```

### Total Parameter Count

```
Global:    917,504 + 114,688 + 448 + 917,504              =   1,950,144
Per layer: 448 + 5×200,704 + 2 + 448 + 3×401,408          =   2,208,642
8 layers:  8 × 2,208,642                                   =  17,669,136
Total:     1,950,144 + 17,669,136                           =  19,619,280
```

19,619,280 parameters × 4 bytes = 78,477,120 bytes ≈ 78.5 MB.

(`nanojanus.html` `loadWeightsPEN7()` lines ~822–877)

### C Binary Format (different)

The C implementations use a different format:
- Magic: `0x4A414E55` ("JANU")
- Stores dual weight matrices (A + B)
- Includes MetaJanus state, AMLState, chambers, Chuck optimizer state
- No PEN7 header structure

(`janus.c` `save_model()` / `load_model()` lines ~1357–1406)

---

## 12. Differences from C Implementations

### Architecture Comparison

| Feature | `nanojanus.html` / `nanojanus.py` | C files (`janus.c` etc.) |
|---------|-----------------------------------|--------------------------|
| **Language** | JavaScript (browser) / Python (CLI) | C (compiled) |
| **Parameters** | 19.6M (single) | Varies: 6.5M–37.6M (×2 dual) |
| **Attention** | QKV + RRPRAM (2-way) | QKV + RRPRAM + Janus (3-way) |
| **Gate** | Per-layer, 2 scalars | Per-head, H×3 logits |
| **Wj matrix** | Not present | E×E per block |
| **Weight matrices** | Single set | Dual (A/B) blended by calendar |
| **Position encoding** | RoPE (θ=10000) | Learned positional embeddings |
| **Vocab level** | BPE→Word (2048→1984+) | Char (256) or BPE (2048) |
| **RRPRAM shape** | DIM×DIM | H×DIM×MAX_T (per head) |
| **RRPRAM values** | Shared with QKV (V) | Separate Wvr projection |
| **Dario overlay** | Full 7-force | Full in janus.c, partial in others |
| **Kuramoto chambers** | 6 chambers | 6 chambers |
| **Calendar drift** | Identical | Identical |
| **Optimizer** | AdamW (`nanojanus.py` only) | Chuck optimizer |
| **Training** | Supported (`nanojanus.py` `train_on_text()`) | Full forward/backward |
| **Weight format** | PEN7 | Custom binary (magic 0x4A414E55) |
| **Wormhole** | Visual indicator only | Actual step skipping |

### C File Comparison

| File | Lines | Vocab | Attention | Dual | BPE | Dario |
|------|-------|-------|-----------|------|-----|-------|
| `janus.c` | ~1545 | 256 (char) | QKV+RRPRAM+Janus | ✓ | ✗ | ✓ |
| `janus-hybrid.c` | ~1088 | 2048+256 | QKV+RRPRAM+Janus | ✓ | ✓ | ✗ |
| `janus-bpe.c` | ~632 | 2048 | QKV+RRPRAM+Janus | ✓ | ✓ | ✗ |
| `metajanus.c` | ~926 | 256 (char) | Janus only (pure) | ✓ | ✗ | ✗ |
| `resonance-janus-bpe.c` | ~771 | configurable | QKV+RRPRAM+Janus | ✓ | ✓ | ✓ |

### Notable C-Only Features

- **Dual weight matrices**: Model A and Model B blended by calendar state
  (`blend_alpha = 0.5 + 0.3·(cal_d − 0.5) − 0.2·debt + 0.1·meta_d`)
- **Chuck optimizer**: Adam-like optimizer with window-based damping
  (`β1=0.9, β2=0.999, ε=1e-8, window=16`)
- **Backward pass**: Full gradient computation and weight updates
- **GGUF export**: `janus.c` can export to GGUF format for external tools
- **metajanus.c pure mode**: Janus self-resonance as the sole attention
  mechanism (no QKV, no RRPRAM) — proving Janus attention is sufficient alone
- **Configurable depth**: `resonance-janus-bpe.c` scales all dimensions via
  a `cfg_from_depth()` function

### Shared Constants (C files)

All C files share: `DIM=384, HEADS=4, HEAD_DIM=96, BLOCKS=12, MLP_DIM=768,
MAX_T=256`. These differ from `nanojanus.html` / `nanojanus.py` (`DIM=448,
N_HEADS=7, HEAD_DIM=64, N_LAYERS=8, HDIM=896`).

### nanojanus.py: Training Support

`nanojanus.py` is the Python CLI equivalent of `nanojanus.html`. Same
architecture (QKV + RRPRAM, 2-way gate, PEN7 weights) but adds training:

**Key functions:**
- `forward(bpe_ids)` (line ~541): Full 8-layer forward pass
- `train_on_text(text, total_steps, lr, wd)` (line ~1023): BPE next-token
  prediction with analytical backward pass
- `save_weights(path)` (line ~1267): PEN7 binary export
- `load_weights(path)` (line ~1315): PEN7 binary import
- `dario_overlay(logits, ...)` (line ~727): 7-force overlay
- `extract_key(text)` (line ~836): Origin word selection
- `run_chain(user_text)` (line ~857): Bidirectional generation loop

**Optimizer**: AdamW (β1=0.9, β2=0.999, ε=1e-8, line ~1013). The
`train_on_text()` docstring (line ~1027) explicitly states `"No Chuck."` — the
Chuck optimizer is C-only (`janus.c` lines ~464–477).

**Training loop** (`train_on_text()`, line ~1023):
1. Random context window (4–32 BPE tokens)
2. Forward pass through 8-layer Resonance engine
3. Cross-entropy loss on next-token prediction
4. Analytical gradient computation for tok_emb, lm_head, final_norm,
   and per-layer weights
5. AdamW update with weight decay

**Save format**: PEN7 binary (magic `0x50454E37`), identical header to
`nanojanus.html` `loadWeightsPEN7()`. Includes co-occurrence maps, bigram
data, and MetaJanus birth state after the weight payload.

---

## Cross-Reference: CASCADE01.md Consistency

CASCADE01.md states:

> **NanoJanus** (19.6M params, Resonance + Janus self-resonance)
> Same base as Penelope (8 layers, DIM=448, 7 heads, SwiGLU, RRPRAM) PLUS
> Janus self-resonance attention.

The 19.6M parameter count, 8 layers, DIM=448, and 7 heads match
`nanojanus.html` and `nanojanus.py`. However, the "PLUS Janus self-resonance"
is aspirational — `nanojanus.html` and `nanojanus.py` implement QKV + RRPRAM
(2-way gate) without the Wj matrix. The full 3-way attention (including Janus
self-resonance) exists only in the C implementations, which use different
dimensions (DIM=384, HEADS=4).

**Status**: This is a documentation-vs-implementation gap, not a bug.
CASCADE01.md describes the intended architecture; `nanojanus.html` /
`nanojanus.py` implement the operational subset needed for the daily cascade.
If Janus self-resonance is added in the future, the PEN7 weight format would
need a Wj matrix per layer and the gate would expand from 2 to 3 values. This
file (`labs/`) does not modify CASCADE01.md — that lives in
`ariannamethod/ariannamethod`.

CASCADE01.md's daily cycle timing (NanoJanus at 06:30 UTC), input sources
(Haiku + Penelope + Molequla), output format (12 bidirectional words), and
health criteria ("produces up to 12 words, origin word identified") are
consistent with the implementation.

---

_These notes describe the architecture as implemented in the repository.
No source code, weights, or vocabulary files were modified._

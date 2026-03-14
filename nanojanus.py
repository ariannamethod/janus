#!/usr/bin/env python3
"""
NanoJanus — Bi-directional associative reasoning in the terminal.

Python port of nanojanus.html's architecture:
  - Word-level output with BPE-like internal embeddings
  - Dual weight matrices blended by calendar drift
  - 12 bi-directional steps (forward + backward)
  - Janus self-resonance attention
  - Dario equation for scoring (7-force)
  - Prophecy, Destiny, Prophecy Debt, Wormhole
  - MetaJanus birth date mathematical "self"
  - Kuramoto chambers (6 coupled oscillators)
  - Chuck optimizer for training
  - RRPRAM-style per-step weight matrices (Wr, RMSNorm, SwiGLU)
"""

import argparse
import math
import os
import pickle
import random
import re
import sys
import time

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
STEPS = 12
DIM = 64
HDIM = 128

# ═══════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════
SUFFIXES = [
    'ing', 'tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ful',
    'less', 'ous', 'ive', 'ity', 'ary', 'ery', 'ory', 'al',
    'ly', 'er', 'ed', 'es', 'en', 's',
]

WORDS = []
VOCAB_LENS = []
NWORDS = 0


def load_vocabulary(vocab_path):
    """Load vocabulary from nanojanus.txt (one word per line)."""
    global WORDS, VOCAB_LENS, NWORDS
    with open(vocab_path, 'r') as f:
        WORDS = [line.strip() for line in f if line.strip()]
    VOCAB_LENS = [len(w) for w in WORDS]
    NWORDS = len(WORDS)


def try_stem(word):
    """Stage 2 tokenizer: try stripping known suffixes to find a vocab match."""
    for suf in SUFFIXES:
        if len(word) > len(suf) + 2 and word.endswith(suf):
            stem = word[:-len(suf)]
            if stem in WORDS:
                return WORDS.index(stem)
    return -1


def greedy_vocab_match(word):
    """Stage 3 tokenizer: greedy longest-match BPE decomposition."""
    wlen = len(word)
    ids = []
    pos = 0
    while pos < wlen and len(ids) < 8:
        best = -1
        best_len = 0
        for v in range(NWORDS):
            vl = VOCAB_LENS[v]
            if vl <= best_len or vl > wlen - pos:
                continue
            if word[pos:pos + vl] == WORDS[v]:
                best = v
                best_len = vl
        if best >= 0 and best_len >= 3:
            ids.append(best)
            pos += best_len
        else:
            pos += 1
    return ids


# ═══════════════════════════════════════════════════════════════
# CALENDAR DRIFT (Metonic cycle, Gregorian vs Hebrew)
# ═══════════════════════════════════════════════════════════════
AM_ANNUAL_DRIFT = 11.25
AM_GREGORIAN_YEAR = 365.25
AM_METONIC_YEARS = 19
AM_METONIC_LEAPS = 7
AM_MAX_UNCORRECTED = 33.0
METONIC_LEAP_YEARS = [3, 6, 8, 11, 14, 17, 19]

# Epoch: 1 Tishrei 5785 = October 3, 2024, 12:00:00 UTC
EPOCH_TIMESTAMP = 1727956800.0  # 2024-10-03 12:00:00 UTC


def calendar_days_since_epoch():
    return int((time.time() - EPOCH_TIMESTAMP) / 86400)


def calendar_cumulative_drift(days):
    years = days / AM_GREGORIAN_YEAR
    base_drift = years * AM_ANNUAL_DRIFT
    full_cycles = int(years / AM_METONIC_YEARS)
    corrections = full_cycles * AM_METONIC_LEAPS * 30
    partial = years % AM_METONIC_YEARS
    yic = int(partial) + 1
    for i in range(AM_METONIC_LEAPS):
        if METONIC_LEAP_YEARS[i] <= yic:
            corrections += 30
    return base_drift - corrections


def calendar_dissonance(days):
    drift = calendar_cumulative_drift(days)
    raw = abs(drift % AM_MAX_UNCORRECTED) / AM_MAX_UNCORRECTED
    return max(0.0, min(1.0, raw))


# ═══════════════════════════════════════════════════════════════
# METAJANUS — birth date snapshot
# ═══════════════════════════════════════════════════════════════
class MetaJanus:
    def __init__(self):
        d = calendar_days_since_epoch()
        self.birth_days = d
        self.birth_drift = calendar_cumulative_drift(d)
        self.birth_dissonance = calendar_dissonance(d)
        self.birth_time = time.time()
        self.prophecy_accuracy = 0.5
        self.total_predictions = 0


META = None  # initialized after vocabulary loads


def personal_dissonance():
    now_drift = calendar_cumulative_drift(calendar_days_since_epoch())
    return max(0.0, min(1.0, abs(now_drift - META.birth_drift) / AM_MAX_UNCORRECTED))


# ═══════════════════════════════════════════════════════════════
# AML PHYSICS
# ═══════════════════════════════════════════════════════════════
prophecy_debt = 0.0
destiny_bias = 0.1
wormhole = 0.02
resonance_field = 0.5
trauma = 0.0


def compute_prophecy_debt(scores, chosen_idx):
    if not scores:
        return 0.0
    mx = max(s['score'] for s in scores)
    chosen = scores[chosen_idx]['score'] if chosen_idx < len(scores) else 0.0
    diff = mx - chosen
    return diff / (diff + 1) if diff > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# KURAMOTO CHAMBERS (6 coupled oscillators)
# ═══════════════════════════════════════════════════════════════
CH_FEAR = 0
CH_LOVE = 1
CH_RAGE = 2
CH_VOID = 3
CH_FLOW = 4
CH_COMPLEX = 5
CH_N = 6

chambers = [0.0] * CH_N
CH_DECAY = [0.95, 0.95, 0.93, 0.96, 0.94, 0.97]


def update_chambers(step_idx):
    global chambers
    depth = step_idx / STEPS
    if depth < 0.33:
        chambers[CH_FLOW] += 0.05
    elif depth < 0.66:
        chambers[CH_FEAR] += 0.04
    else:
        chambers[CH_VOID] += 0.05
    if depth > 0.75:
        chambers[CH_COMPLEX] += 0.03
    if trauma > 0.3:
        chambers[CH_RAGE] += 0.04
    K = 0.02
    old = list(chambers)
    for i in range(CH_N):
        for j in range(CH_N):
            if i != j:
                chambers[i] += K * math.sin(old[j] - old[i])
        chambers[i] = max(0.0, min(1.0, chambers[i] * CH_DECAY[i]))


# ═══════════════════════════════════════════════════════════════
# DUAL WEIGHT MATRICES + RRPRAM per-step weights
# ═══════════════════════════════════════════════════════════════
embedA = []
embedB = []
step_wr_a = []
step_wr_b = []
step_rms_a = []
step_rms_b = []
step_gate_a = []
step_up_a = []
step_down_a = []
step_gate_b = []
step_up_b = []
step_down_b = []


def _rand_vec(size, scale):
    return [(random.random() - 0.5) * scale for _ in range(size)]


def _ones_vec(size):
    return [1.0] * size


def init_weights():
    """Initialize all weight matrices (dual embeddings + RRPRAM per-step)."""
    global embedA, embedB
    global step_wr_a, step_wr_b, step_rms_a, step_rms_b
    global step_gate_a, step_up_a, step_down_a
    global step_gate_b, step_up_b, step_down_b

    scale = math.sqrt(2.0 / DIM) * 0.02
    embedA = _rand_vec(NWORDS * DIM, scale)
    embedB = _rand_vec(NWORDS * DIM, scale)

    step_wr_a = []
    step_wr_b = []
    step_rms_a = []
    step_rms_b = []
    step_gate_a = []
    step_up_a = []
    step_down_a = []
    step_gate_b = []
    step_up_b = []
    step_down_b = []

    for _ in range(STEPS):
        step_wr_a.append(_rand_vec(DIM * DIM, scale))
        step_wr_b.append(_rand_vec(DIM * DIM, scale))
        step_rms_a.append(_ones_vec(DIM))
        step_rms_b.append(_ones_vec(DIM))
        step_gate_a.append(_rand_vec(DIM * HDIM, scale))
        step_up_a.append(_rand_vec(DIM * HDIM, scale))
        step_down_a.append(_rand_vec(HDIM * DIM, scale))
        step_gate_b.append(_rand_vec(DIM * HDIM, scale))
        step_up_b.append(_rand_vec(DIM * HDIM, scale))
        step_down_b.append(_rand_vec(HDIM * DIM, scale))


# ═══════════════════════════════════════════════════════════════
# BLEND (calendar-driven dual matrix interpolation)
# ═══════════════════════════════════════════════════════════════
def get_blend_alpha():
    cd = calendar_dissonance(calendar_days_since_epoch())
    pd = personal_dissonance()
    return max(0.0, min(1.0, 0.5 + 0.3 * (cd - 0.5) - 0.2 * prophecy_debt + 0.1 * pd))


def blended_embed(word_idx):
    a = get_blend_alpha()
    b = 1.0 - a
    out = [0.0] * DIM
    base = word_idx * DIM
    for i in range(DIM):
        out[i] = a * embedA[base + i] + b * embedB[base + i]
    return out


# ═══════════════════════════════════════════════════════════════
# CO-OCCURRENCE + BIGRAMS
# ═══════════════════════════════════════════════════════════════
cooc = {}
bigrams = {}


def update_cooc(w1, w2):
    key = (min(w1, w2), max(w1, w2))
    cooc[key] = cooc.get(key, 0) + 1


def get_cooc(w1, w2):
    return cooc.get((min(w1, w2), max(w1, w2)), 0)


def update_bigram(prev, nxt):
    key = (prev, nxt)
    bigrams[key] = bigrams.get(key, 0) + 1


def get_bigram(prev, nxt):
    return bigrams.get((prev, nxt), 0)


# ═══════════════════════════════════════════════════════════════
# TOKENIZER (3-stage: exact → stem → greedy BPE decomposition)
# ═══════════════════════════════════════════════════════════════
def tokenize_text(text):
    words = re.sub(r'[^a-z\s]', '', text.lower()).split()
    ids = []
    for w in words:
        if not w:
            continue
        # Stage 1: exact match
        if w in WORDS:
            ids.append(WORDS.index(w))
            continue
        # Stage 2: stem matching
        stem_idx = try_stem(w)
        if stem_idx >= 0:
            ids.append(stem_idx)
            continue
        # Stage 3: greedy longest-match BPE decomposition
        sub = greedy_vocab_match(w)
        ids.extend(sub)
    return ids


# ═══════════════════════════════════════════════════════════════
# DARIO EQUATION — 7-force word scoring
# ═══════════════════════════════════════════════════════════════
def silu(x):
    return x / (1.0 + math.exp(-x)) if x > -20 else 0.0


def dario_score(candidate_idx, context, prev_word, step_idx, direction):
    alpha_mod = 1.0 + 0.3 * chambers[CH_LOVE] - 0.2 * chambers[CH_RAGE] + 0.1 * chambers[CH_FLOW]
    gamma_mod = 1.0 + 0.4 * chambers[CH_VOID] + 0.2 * chambers[CH_COMPLEX]

    # B: bigram transition
    B = 0.0
    if prev_word >= 0:
        B = math.log(1 + get_bigram(prev_word, candidate_idx)) * 4

    # H: Hebbian co-occurrence
    H = 0.0
    for c in context:
        H += math.log(1 + get_cooc(c, candidate_idx))
    H = H / (len(context) + 1)

    # F: Prophecy fulfillment (scaled by debt)
    F = prophecy_debt * (1.0 + random.random() * 0.5)

    # A: Destiny attraction
    A = destiny_bias * gamma_mod * 0.5

    # Direction modulation: backward steps more conservative
    dir_mod = 0.8 if direction == -1 else 1.2

    # SwiGLU gating through resonance
    gate = 1.0 / (1.0 + math.exp(-(resonance_field - 0.5) * 4))
    h_g = silu(gate * 2)
    f_g = silu(gate * 1.5)

    # Calendar dissonance modulation
    cal_mod = 1.0 + 0.2 * calendar_dissonance(calendar_days_since_epoch())

    tau = 0.7 + step_idx * 0.06 + prophecy_debt * 0.3
    score = (B + alpha_mod * 3 * H * h_g + 2 * F * f_g + A) * dir_mod * cal_mod / tau

    return score + (random.random() - 0.5) * 0.3 * (1 - step_idx / STEPS)


# ═══════════════════════════════════════════════════════════════
# WORD SELECTION (top-k sampling)
# ═══════════════════════════════════════════════════════════════
def select_word(context, prev_word, step_idx, forbidden, direction):
    scores = []
    for i in range(NWORDS):
        if i in forbidden:
            continue
        scores.append({
            'word': WORDS[i],
            'idx': i,
            'score': dario_score(i, context, prev_word, step_idx, direction),
        })
    scores.sort(key=lambda s: s['score'], reverse=True)
    top_k = min(8, len(scores))
    top = scores[:top_k]
    for s in top:
        s['score'] = max(0.0, s['score'])
    total = sum(s['score'] for s in top)
    if total < 1e-10:
        return top[0]
    r = random.random() * total
    cum = 0.0
    for s in top:
        cum += s['score']
        if cum >= r:
            return s
    return top[0]


# ═══════════════════════════════════════════════════════════════
# BI-DIRECTIONAL REASONING (12-step forward + backward)
# ═══════════════════════════════════════════════════════════════
def extract_key(text):
    """Find the most 'charged' word in the prompt."""
    ids = tokenize_text(text)
    if not ids:
        return random.randint(0, NWORDS - 1)
    best = ids[0]
    best_score = 0
    for wid in ids:
        sc = 0
        for key, val in cooc.items():
            if key[0] == wid or key[1] == wid:
                sc += val
        if sc > best_score:
            best_score = sc
            best = wid
    return best


def run_chain(user_text):
    """Run the 12-step bi-directional generation chain."""
    global prophecy_debt, chambers

    context = tokenize_text(user_text)
    seed_idx = extract_key(user_text)
    seed_word = WORDS[seed_idx] if seed_idx < NWORDS else 'void'

    # MetaJanus prophecy
    cal_d = calendar_dissonance(calendar_days_since_epoch())
    predicted_entropy = 0.5 + 0.2 * prophecy_debt + 0.1 * cal_d + 0.15 * personal_dissonance()

    # Determine forward/backward split by prophecy debt
    n_backward = max(1, min(STEPS - 1,
                            int(STEPS * (0.3 + 0.4 * prophecy_debt + 0.1 * cal_d))))
    n_forward = STEPS - n_backward

    back_steps = []
    fwd_steps = []
    forbidden = set()

    # Forward steps (future)
    chain = list(context)
    prev_word = seed_idx
    s = 0
    while s < n_forward:
        update_chambers(s)
        is_wormhole = False
        if prophecy_debt < 0.2 and wormhole > 0.1 and random.random() < wormhole:
            is_wormhole = True
            s += random.randint(0, 1)  # skip ahead
        sel = select_word(chain, prev_word, s, forbidden, 1)
        fwd_steps.append({
            'word': sel['word'], 'idx': sel['idx'], 'step': s,
            'wormhole': is_wormhole, 'debt': prophecy_debt,
        })
        chain.append(sel['idx'])
        if prev_word >= 0:
            update_bigram(prev_word, sel['idx'])
        for c in context:
            update_cooc(c, sel['idx'])
        forbidden.add(sel['idx'])
        prophecy_debt = 0.9 * prophecy_debt + 0.1 * compute_prophecy_debt([sel], 0)
        prev_word = sel['idx']
        s += 1

    # Backward steps (past — more exploratory)
    chain = list(context)
    prev_word = seed_idx
    for s in range(n_backward):
        update_chambers(n_forward + s)
        sel = select_word(chain, prev_word, n_forward + s, forbidden, -1)
        back_steps.append({
            'word': sel['word'], 'idx': sel['idx'], 'step': n_forward + s,
            'wormhole': False, 'debt': prophecy_debt,
        })
        chain.append(sel['idx'])
        forbidden.add(sel['idx'])
        prophecy_debt = 0.9 * prophecy_debt + 0.1 * compute_prophecy_debt([sel], 0)
        prev_word = sel['idx']

    # Evaluate MetaJanus prophecy
    all_steps = fwd_steps + back_steps
    avg_debt = sum(x['debt'] for x in all_steps) / STEPS if all_steps else 0
    error = abs(predicted_entropy - avg_debt)
    META.prophecy_accuracy = 0.9 * META.prophecy_accuracy + 0.1 * (1 - error)
    META.total_predictions += 1

    return seed_word, seed_idx, fwd_steps, back_steps


def display_chain(seed_word, fwd_steps, back_steps):
    """Print the bi-directional chain to the terminal."""
    # Backward steps (top, reversed)
    for s in reversed(back_steps):
        wh = ' \u2295WH' if s['wormhole'] else ''
        print(f"  \u2191{s['step']:<3} {s['word']:<16} debt={s['debt']:.2f}{wh}")

    # Origin
    print(f"  \u25cf {seed_word} \u25cf".center(40) + '  [ORIGIN]')

    # Forward steps (bottom)
    for s in fwd_steps:
        wh = ' \u2295WH' if s['wormhole'] else ''
        print(f"  \u2193{s['step']:<3} {s['word']:<16} debt={s['debt']:.2f}{wh}")


def display_metrics():
    """Print drift/dissonance/blend/debt metrics."""
    days = calendar_days_since_epoch()
    drift = calendar_cumulative_drift(days)
    diss = calendar_dissonance(days)
    pd = personal_dissonance()
    blend = get_blend_alpha()
    print(f"\ndrift={drift:.2f}  diss={diss:.3f}  personal={pd:.3f}  "
          f"blend={blend:.3f}  debt={prophecy_debt:.3f}  "
          f"prophecy_acc={META.prophecy_accuracy:.3f}")


# ═══════════════════════════════════════════════════════════════
# TRAINING — word prediction with SGD + Chuck-like modulation
# ═══════════════════════════════════════════════════════════════
def train_on_text(text, total_steps=2000, lr=0.001):
    """Training loop with Chuck optimizer (macro patience, lambda modulation, stagnation noise)."""
    global embedA, embedB

    ids = tokenize_text(text)
    if len(ids) < STEPS + 1:
        print('Error: text too short after tokenization '
              f'({len(ids)} tokens, need at least {STEPS + 1})')
        return

    print(f'Tokenized: {len(ids)} tokens')
    print(f'Training for {total_steps} steps ...')

    # Chuck optimizer state
    best_loss = float('inf')
    patience_counter = 0
    macro_patience = 200
    chuck_lambda = 1.0

    for step in range(total_steps):
        offset = random.randint(0, len(ids) - STEPS - 2)
        loss = 0.0

        for s in range(STEPS):
            ctx_ids = ids[offset:offset + s + 1]
            target = ids[offset + s + 1]

            # Pool context embedding
            ctx = [0.0] * DIM
            for wid in ctx_ids:
                base = wid * DIM
                for d in range(DIM):
                    ctx[d] += embedA[base + d]
            inv = 1.0 / len(ctx_ids)
            for d in range(DIM):
                ctx[d] *= inv

            # RRPRAM: query = ctx @ Wr
            query = [0.0] * DIM
            for i in range(DIM):
                acc = 0.0
                for j in range(DIM):
                    acc += step_wr_a[s][i * DIM + j] * ctx[j]
                query[i] = acc

            # RMSNorm
            ss = sum(q * q for q in query)
            rms_inv = 1.0 / math.sqrt(ss / DIM + 1e-5)
            qn = [step_rms_a[s][i] * query[i] * rms_inv for i in range(DIM)]

            # Logits = E^T @ qn
            logits = [0.0] * NWORDS
            for v in range(NWORDS):
                acc = 0.0
                vbase = v * DIM
                for d in range(DIM):
                    acc += embedA[vbase + d] * qn[d]
                logits[v] = acc

            # Softmax + cross-entropy loss
            mx = max(logits)
            exp_logits = [math.exp(l - mx) for l in logits]
            total = sum(exp_logits)
            probs = [e / total for e in exp_logits]

            p = max(1e-10, probs[target])
            loss -= math.log(p)

            # Gradient: d_logits = probs - one_hot
            d_logits = list(probs)
            d_logits[target] -= 1.0

            # Chuck modulation: lambda adapts to stagnation
            chuck_mod = chuck_lambda

            # Update embeddings (SGD with Chuck modulation)
            for v in range(NWORDS):
                if abs(d_logits[v]) < 1e-6:
                    continue
                vbase = v * DIM
                for d in range(DIM):
                    embedA[vbase + d] -= lr * chuck_mod * d_logits[v] * qn[d]

            # Update Wr gradients
            for i in range(DIM):
                for j in range(DIM):
                    step_wr_a[s][i * DIM + j] -= lr * chuck_mod * 0.01 * qn[i] * ctx[j]

        # Train matrix B on even steps
        if step % 2 == 0:
            off2 = random.randint(0, len(ids) - 2)
            target2 = ids[off2 + 1]
            ctx2 = [0.0] * DIM
            src_base = ids[off2] * DIM
            for d in range(DIM):
                ctx2[d] = embedB[src_base + d]
            for v in range(NWORDS):
                vbase = v * DIM
                sc = sum(embedB[vbase + d] * ctx2[d] for d in range(DIM))
                grad = (-1.0 if v == target2 else 0.0) + 0.001
                for d in range(DIM):
                    embedB[vbase + d] -= lr * grad * ctx2[d] * 0.1

        # Chuck optimizer: macro patience + stagnation noise
        avg_loss = loss / STEPS
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            chuck_lambda = max(0.5, chuck_lambda * 0.99)
        else:
            patience_counter += 1
            if patience_counter > macro_patience:
                # Stagnation noise: perturb learning rate
                chuck_lambda = min(2.0, chuck_lambda * 1.05)
                noise_scale = 0.001 * chuck_lambda
                for idx in range(len(embedA)):
                    embedA[idx] += (random.random() - 0.5) * noise_scale
                patience_counter = 0

        if step % 100 == 0:
            print(f'  step {step:>5}/{total_steps}  loss={avg_loss:.4f}  '
                  f'chuck_λ={chuck_lambda:.3f}  best={best_loss:.4f}')

    print(f'Training complete: {total_steps} steps, final best loss={best_loss:.4f}')


# ═══════════════════════════════════════════════════════════════
# SAVE / LOAD WEIGHTS (pickle)
# ═══════════════════════════════════════════════════════════════
WEIGHTS_FILE = 'nanojanus.weights.pkl'


def save_weights(path=WEIGHTS_FILE):
    data = {
        'embedA': embedA, 'embedB': embedB,
        'step_wr_a': step_wr_a, 'step_wr_b': step_wr_b,
        'step_rms_a': step_rms_a, 'step_rms_b': step_rms_b,
        'step_gate_a': step_gate_a, 'step_up_a': step_up_a,
        'step_down_a': step_down_a,
        'step_gate_b': step_gate_b, 'step_up_b': step_up_b,
        'step_down_b': step_down_b,
        'cooc': cooc, 'bigrams': bigrams,
        'prophecy_debt': prophecy_debt,
        'meta': {
            'prophecy_accuracy': META.prophecy_accuracy,
            'total_predictions': META.total_predictions,
        },
        'nwords': NWORDS,
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Weights saved to {path}')


def load_weights(path=WEIGHTS_FILE):
    global embedA, embedB, prophecy_debt, cooc, bigrams
    global step_wr_a, step_wr_b, step_rms_a, step_rms_b
    global step_gate_a, step_up_a, step_down_a
    global step_gate_b, step_up_b, step_down_b

    if not os.path.exists(path):
        return False
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if data.get('nwords') != NWORDS:
        print(f'Warning: weight file vocab size ({data.get("nwords")}) '
              f'!= current ({NWORDS}), reinitializing')
        return False
    embedA = data['embedA']
    embedB = data['embedB']
    step_wr_a = data['step_wr_a']
    step_wr_b = data['step_wr_b']
    step_rms_a = data['step_rms_a']
    step_rms_b = data['step_rms_b']
    step_gate_a = data['step_gate_a']
    step_up_a = data['step_up_a']
    step_down_a = data['step_down_a']
    step_gate_b = data['step_gate_b']
    step_up_b = data['step_up_b']
    step_down_b = data['step_down_b']
    cooc.update(data.get('cooc', {}))
    bigrams.update(data.get('bigrams', {}))
    prophecy_debt = data.get('prophecy_debt', 0.0)
    meta = data.get('meta', {})
    META.prophecy_accuracy = meta.get('prophecy_accuracy', 0.5)
    META.total_predictions = meta.get('total_predictions', 0)
    print(f'Weights loaded from {path}')
    return True


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def main():
    global META

    parser = argparse.ArgumentParser(
        description='NanoJanus — Bi-directional associative reasoning')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--generate', type=str, metavar='PROMPT',
                       help='Generate a bi-directional chain from prompt text')
    group.add_argument('--train', type=str, metavar='FILE',
                       help='Train on a text file')
    parser.add_argument('--vocab', type=str, default=None,
                        help='Path to vocabulary file (default: nanojanus.txt beside script)')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Training steps (default: 2000)')
    parser.add_argument('--weights', type=str, default=WEIGHTS_FILE,
                        help='Weights file path (default: nanojanus.weights.pkl)')
    args = parser.parse_args()

    # Locate vocabulary file
    vocab_path = args.vocab
    if vocab_path is None:
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'nanojanus.txt')
    if not os.path.exists(vocab_path):
        print(f'Error: vocabulary file not found: {vocab_path}', file=sys.stderr)
        sys.exit(1)

    load_vocabulary(vocab_path)
    META = MetaJanus()
    init_weights()
    load_weights(args.weights)

    if args.generate:
        seed_word, seed_idx, fwd_steps, back_steps = run_chain(args.generate)
        print()
        display_chain(seed_word, fwd_steps, back_steps)
        display_metrics()
        print()

    elif args.train:
        if not os.path.exists(args.train):
            print(f'Error: training file not found: {args.train}', file=sys.stderr)
            sys.exit(1)
        with open(args.train, 'r') as f:
            text = f.read()
        train_on_text(text, total_steps=args.steps)
        save_weights(args.weights)


if __name__ == '__main__':
    main()

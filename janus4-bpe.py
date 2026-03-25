#!/usr/bin/env python3
"""
janus4-bpe.py — 4-attention Janus reference (QKV + RRPRAM + Echo + Temporal Delta)
Includes tokenizer, model, training, and generation in one file.
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteBPE:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.merges: List[tuple[int, int, int]] = []

    def learn(self, data: bytes, max_merges: int | None = None):
        if max_merges is None:
            max_merges = self.vocab_size - 256
        tokens = list(data)
        for m in range(max_merges):
            counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            (a, b), c = max(counts.items(), key=lambda x: x[1])
            if c < 2 or 256 + len(self.merges) >= self.vocab_size:
                break
            nid = 256 + len(self.merges)
            self.merges.append((a, b, nid))
            out = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                    out.append(nid)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out

    def encode(self, data: bytes) -> List[int]:
        out = list(data)
        for a, b, r in self.merges:
            i, nxt = 0, []
            while i < len(out):
                if i + 1 < len(out) and out[i] == a and out[i + 1] == b:
                    nxt.append(r)
                    i += 2
                else:
                    nxt.append(out[i])
                    i += 1
            out = nxt
        return out

    def _decode_tok(self, tok: int) -> bytes:
        if tok < 256:
            return bytes([tok])
        for a, b, r in reversed(self.merges):
            if r == tok:
                return self._decode_tok(a) + self._decode_tok(b)
        return b"?"

    def decode(self, toks: List[int]) -> str:
        return b"".join(self._decode_tok(t) for t in toks).decode("utf-8", errors="ignore")


class Janus4Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, max_t: int):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.max_t = max_t

        self.rms1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.rrpram_k = nn.Linear(dim, dim, bias=False)
        self.echo_proj = nn.Linear(dim, dim, bias=False)
        self.delta_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(heads, 4))
        self.out = nn.Linear(dim, dim, bias=False)

        self.rms2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(), nn.Linear(mlp_dim, dim))

    def _causal_mask(self, T: int, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        xn = self.rms1(x)

        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        mask = self._causal_mask(T, x.device)
        scale = 1.0 / math.sqrt(self.head_dim)

        # 1) QKV attention
        s_qkv = (q @ k.transpose(-2, -1)) * scale
        s_qkv = s_qkv.masked_fill(mask, -1e9)
        y_qkv = F.softmax(s_qkv, dim=-1) @ v

        # 2) RRPRAM-like memory attention (projected K, shared V)
        rk = self.rrpram_k(xn).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        s_rr = (q @ rk.transpose(-2, -1)) * scale
        s_rr = s_rr.masked_fill(mask, -1e9)
        y_rr = F.softmax(s_rr, dim=-1) @ v

        # 3) Echo attention
        echo = self.echo_proj(xn)
        echo = echo.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        s_echo = (q @ echo.transpose(-2, -1)) * scale
        s_echo = s_echo.masked_fill(mask, -1e9)
        y_echo = F.softmax(s_echo, dim=-1) @ echo

        # 4) Temporal Delta attention (new)
        prev = torch.roll(xn, shifts=1, dims=1)
        prev[:, 0] = 0
        delta = self.delta_proj(xn - prev)
        delta = delta.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        energy = delta.abs().mean(dim=-1, keepdim=True)  # B,H,T,1
        s_delta = (energy @ energy.transpose(-2, -1)).squeeze(-1)
        s_delta = s_delta.unsqueeze(1).repeat(1, self.heads, 1, 1)
        dist = torch.arange(T, device=x.device)
        decay = 1.0 / (1.0 + (dist[None, :] - dist[:, None]).abs().float())
        s_delta = s_delta * decay
        s_delta = s_delta.masked_fill(mask, -1e9)
        y_delta = F.softmax(s_delta, dim=-1) @ delta

        g = F.softmax(self.gate, dim=-1).view(1, self.heads, 1, 4)
        y = (
            g[..., 0:1] * y_qkv
            + g[..., 1:2] * y_rr
            + g[..., 2:3] * y_echo
            + g[..., 3:4] * y_delta
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out(y)
        x = x + self.mlp(self.rms2(x))
        return x


class Janus4(nn.Module):
    def __init__(self, vocab: int, max_t: int = 128, dim: int = 256, heads: int = 4, blocks: int = 6):
        super().__init__()
        self.max_t = max_t
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_t, dim)
        self.blocks = nn.ModuleList([Janus4Block(dim, heads, dim * 2, max_t) for _ in range(blocks)])
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x))


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


def make_batch(tokens: List[int], seq: int, bs: int, device: str) -> Batch:
    x, y = [], []
    hi = len(tokens) - seq - 1
    for _ in range(bs):
        i = random.randint(0, hi)
        x.append(tokens[i : i + seq])
        y.append(tokens[i + 1 : i + seq + 1])
    return Batch(torch.tensor(x, device=device), torch.tensor(y, device=device))


def train(args):
    with open(args.train, "rb") as f:
        raw = f.read()

    bpe = ByteBPE(vocab_size=args.vocab)
    bpe.learn(raw)
    tokens = bpe.encode(raw)
    print(f"[janus4-bpe.py] bytes={len(raw)} tokens={len(tokens)} merges={len(bpe.merges)}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = Janus4(vocab=args.vocab, max_t=args.seq, dim=args.dim, heads=args.heads, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    for step in range(1, args.steps + 1):
        b = make_batch(tokens, args.seq, args.batch, device)
        logits = model(b.x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), b.y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        best = min(best, float(loss))
        if step == 1 or step % args.log_every == 0:
            print(f"step {step:4d}/{args.steps} loss={float(loss):.4f} best={best:.4f}")

    if args.save:
        torch.save({"model": model.state_dict(), "bpe_merges": bpe.merges, "cfg": vars(args)}, args.save)
        print(f"saved: {args.save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

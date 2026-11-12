"""The gradient-descent baseline: LoRA on every MLP projection, Adam on the same loss and rows as the direct edit,
checkpoint chosen on the same kept-back rows, then merged into the weights."""
from __future__ import annotations

import gc
import math
import re
import time
from typing import Dict, Sequence

import torch

from wspd.edit import Bench
from wspd.reader import sync


class LoRA:
    """out += (alpha / rank) * (x @ A.T) @ B.T on gate_proj/up_proj/down_proj of every layer, by forward hooks.
    A starts random and B at zero, so the model starts unchanged; both train in fp32 while the weights stay frozen."""

    def __init__(self, model, rank: int = 16, alpha: float = 32.0,
                 targets: Sequence[str] = ("gate_proj", "up_proj", "down_proj"), seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.scale = alpha / rank
        self.mods, self.params, self.handles = [], [], []
        pat = re.compile(r"layers\.\d+\.mlp\.(%s)$" % "|".join(targets))
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and pat.search(name):
                dev = mod.weight.device
                A = (torch.randn(rank, mod.in_features, generator=g) / math.sqrt(mod.in_features)).to(dev)
                B = torch.zeros(mod.out_features, rank, device=dev)
                A.requires_grad_(True); B.requires_grad_(True)
                self.mods.append((mod, A, B)); self.params += [A, B]
                self.handles.append(mod.register_forward_hook(self._hook(A, B)))

    def _hook(self, A, B):
        def h(mod, inp, out):
            return out + (self.scale * ((inp[0].float() @ A.T) @ B.T)).to(out.dtype)
        return h

    def n_params(self) -> int:
        return sum(p.numel() for p in self.params)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def bake(self) -> Dict:
        dev = self.mods[0][0].weight.device
        changed = total = 0
        with torch.no_grad():
            sync(dev); t = time.perf_counter()
            for mod, A, B in self.mods:
                W = mod.weight
                new = (W.float() + self.scale * (B @ A)).to(W.dtype)
                changed += int((new != W).sum().item()); total += W.numel()
                W.copy_(new)
            sync(dev); ms = (time.perf_counter() - t) * 1000
        self.remove()
        return {"bake_ms": ms, "weights_in_targets": total, "weights_changed": changed, "matrices": len(self.mods)}


def train_lora(bench: Bench, lora: LoRA, steps: int, lr: float, batch: int, guard_weight: float, eval_every: int,
               chunk: int = 8, log=None, seed: int = 0, forgetting: bool = False) -> Dict:
    """Adam on the LoRA matrices: the teacher gap over minibatches of the training rows plus guard_weight x (guard KL
    + post-answer KL), the same loss the direct edit solves for. The checkpoint kept is the best on the kept-back
    rows (agreement, then gap), the same rule the direct edit's rounds use."""
    opt = torch.optim.Adam(lora.params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    scale = 1024.0
    hist, evals, skipped = [], [], 0
    best_key, best_step, best_state = None, 0, [p.detach().clone() for p in lora.params]
    t_start = time.perf_counter()
    for s in range(1, steps + 1):
        sync(bench.device); t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        rows = torch.randperm(bench.n["train"], generator=g)[:batch].to(bench.device)
        total = 0.0
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            gq = bench.gap("train", r) * (len(r) / len(rows))
            (gq * scale).backward(); total += gq.item()
        kl = 0.0
        if guard_weight > 0:
            k = bench.guard_kl_grad()
            (guard_weight * k * scale).backward(); kl = k.item()
            rows = torch.randperm(bench.n["train"], generator=g)[:batch].to(bench.device)
            for i in range(0, len(rows), chunk):
                r = rows[i:i + chunk]
                pk = bench.post_kl("train", r) * (len(r) / len(rows))
                (guard_weight * pk * scale).backward(); kl += pk.item()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in lora.params):
            skipped += 1; scale /= 2; continue
        for p in lora.params:
            p.grad.div_(scale)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(bench.device)
        if bench.device.type == "mps":      # the caching allocator otherwise grows past physical memory and steps slow 3-5x
            gc.collect(); torch.mps.empty_cache()
        hist.append({"step": s, "ms": (time.perf_counter() - t) * 1000, "gap": total, "guard_kl": kl})
        if s % eval_every == 0 or s == steps:
            m = bench.evaluate(("val",), forgetting=forgetting)
            evals.append({"step": s, **m})
            key = (m["val_agree"], -m["val_gap"])
            if best_key is None or key > best_key:
                best_key, best_step = key, s
                best_state = [p.detach().clone() for p in lora.params]
            if log:
                log(s, hist[-1], m)
    with torch.no_grad():
        for p, b in zip(lora.params, best_state):
            p.copy_(b)
    return {"hist": hist, "evals": evals, "best_step": best_step, "skipped": skipped,
            "train_seconds": time.perf_counter() - t_start}

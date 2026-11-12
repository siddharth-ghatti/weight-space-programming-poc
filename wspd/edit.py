"""The direct edit: distil the teacher's calibrated probabilities into the reader's own weights with closed-form
column edits of one `mlp.down_proj` (no gradient descent), as in the fast-edits notebook, generalized to an
8-way choice. One round: read keys and final residuals at the answer position (forward passes), solve
dW_S = R^T K_S (K_S^T K_S + lambda n C_SS)^-1 (MEMIT normal equations restricted to the columns S), line-search a
dose on the kept-back rows with a guard-KL cap, repeat until the kept-back rows stop improving; then bake."""
from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional, Sequence

import torch

from wspd.common import (CAPITALS, CONTROL_TEMPLATE, GUARD, HOLDOUT, K, ece, parse_word, prompt, selective_accuracy)
from wspd.reader import Student, down_proj, sync


class _Stop(Exception):
    pass


class ColumnEdit:
    """W[:, cols] += alpha * D (+ beta * R), applied in the forward pass with D in fp32, so doses and directions can
    be tried without touching the weight until bake()."""

    def __init__(self, model, layer: int, cols: Sequence[int], module: Optional[torch.nn.Linear] = None):
        self.mod = module if module is not None else down_proj(model, layer)
        W = self.mod.weight
        self.layer = layer
        self.cols = torch.as_tensor(list(cols), dtype=torch.long, device=W.device)
        self.D = torch.zeros(W.shape[0], len(self.cols), dtype=torch.float32, device=W.device)
        self.alpha = 1.0
        self.R: Optional[torch.Tensor] = None
        self.beta = 0.0
        self.handle = self.mod.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0 and (self.R is None or self.beta == 0.0):
            return out
        D = self.alpha * self.D
        if self.R is not None and self.beta != 0.0:
            D = D + self.beta * self.R
        x = inp[0][..., self.cols].float()
        return out + (x @ D.T).to(out.dtype)

    def remove(self) -> None:
        self.handle.remove()

    def bake(self) -> Dict:
        """Write D into the weight and drop the hook: from here the reader carries the edit itself."""
        W = self.mod.weight
        with torch.no_grad():
            before = W[:, self.cols].clone()
            sync(W.device); t = time.perf_counter()
            W[:, self.cols] = (W[:, self.cols].float() + self.alpha * self.D).to(W.dtype)
            sync(W.device); ms = (time.perf_counter() - t) * 1000
            changed = int((W[:, self.cols] != before).sum().item())
        self.remove()
        return {"bake_ms": ms, "weights_in_edit": int(before.numel()), "weights_changed": changed, "backup": before}

    def restore(self, backup: torch.Tensor) -> None:
        with torch.no_grad():
            self.mod.weight[:, self.cols] = backup


class Bench:
    """One experiment's state: the student, every split encoded with its teacher targets, the unedited baselines."""

    def __init__(self, student: Student, splits: Dict[str, List[dict]], variant: str = "words", chunk: int = 32,
                 kc: Optional[Dict[int, tuple]] = None, post_splits: Sequence[str] = ("train", "val")):
        self.s = student
        self.device = student.device
        self.variant = variant
        self.chunk = chunk
        self.data = {}
        for split, rows in splits.items():
            enc = student.encode([prompt(r["text"], variant) for r in rows])
            T = torch.tensor([r["teacher"] for r in rows], dtype=torch.float32, device=self.device)
            y = torch.tensor([r["gold"] for r in rows], device=self.device)
            self.data[split] = (enc, T, y)
        self.n = {split: len(rows) for split, rows in splits.items()}
        self.spelling = student.choose_spelling(self.data["train"][0])
        # After the answer: the prompt plus the teacher's answer word. The next-token distribution there is the
        # unedited model's own; both arms are held to it so neither learns to keep answering.
        self.post = {}
        with torch.no_grad():
            for split, rows in splits.items():
                if split not in post_splits:
                    continue
                words = [student.words[int(torch.tensor(r["teacher"]).argmax())] for r in rows]
                enc = student.encode([prompt(r["text"], variant) + student.answer_prefix + w for r, w in zip(rows, words)])
                tops = []
                for i in range(0, len(rows), chunk):
                    lp = student.last_logp(enc, torch.arange(i, min(i + chunk, len(rows)), device=self.device))
                    t = lp.topk(64, dim=-1); tops.append((t.indices, t.values))
                self.post[split] = (enc, torch.cat([a for a, _ in tops]), torch.cat([b for _, b in tops]))
            self.guard = student.encode(GUARD)
            self.holdout = student.encode(HOLDOUT)
            self.control = student.encode([CONTROL_TEMPLATE.format(c=k) for _, k in CAPITALS])
            self.guard_lp = student.text_logp(self.guard)
            self.holdout_lp = student.text_logp(self.holdout)
            self.holdout_tokens = self._holdout_targets()
        self._kc: Dict[int, tuple] = kc if kc is not None else {}   # keys/covariance per layer; shareable between benches
        self._norm32 = None

    def _holdout_targets(self) -> torch.Tensor:
        ids, attn, _ = self.holdout
        nxt = torch.full_like(ids, -1)
        nxt[:, :-1] = ids[:, 1:]
        nxt[attn == 0] = -1
        nxt[torch.arange(len(ids)), attn.sum(1) - 1] = -1
        return nxt[attn.bool()]

    # --- losses
    def gap(self, split: str, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-entropy from the teacher's probabilities to the student's full-vocabulary probabilities of the option
        words, minus the teacher's entropy: zero only when the student puts exactly the teacher's probabilities on the
        option words and nothing elsewhere."""
        enc, T, _ = self.data[split]
        if rows is not None:
            T = T[rows]
        lp = self.s.last_logp(enc, rows)[:, self.s.word_ids]
        ent = (T * T.clamp_min(1e-12).log()).sum(-1)
        return (ent - (T * lp).sum(-1)).mean()

    @torch.no_grad()
    def gap_value(self, split: str, rows: Optional[torch.Tensor] = None) -> float:
        if rows is None:
            rows = torch.arange(self.n[split], device=self.device)
        n, c = len(rows), self.chunk
        return sum(self.gap(split, rows[i:i + c]).item() * min(c, n - i) / n for i in range(0, n, c))

    def post_kl(self, split: str, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc, ids, base = self.post[split]
        if rows is not None:
            ids, base = ids[rows], base[rows]
        lp = self.s.last_logp(enc, rows)
        return (base.exp() * (base - lp.gather(1, ids))).sum(-1).mean()

    @torch.no_grad()
    def post_kl_value(self, split: str) -> float:
        n, c = len(self.post[split][1]), self.chunk
        return sum(self.post_kl(split, torch.arange(i, min(i + c, n), device=self.device)).item() * min(c, n - i) / n
                   for i in range(0, n, c))

    def guard_kl_grad(self) -> torch.Tensor:
        lp = self.s.text_logp(self.guard)
        return (self.guard_lp.exp() * (self.guard_lp - lp)).sum(-1).mean()

    @torch.no_grad()
    def guard_kl(self) -> float:
        return self.guard_kl_grad().item()

    # --- measurement at the answer position
    @torch.no_grad()
    def option_probs(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-vocab log-probs of the option words [n, K], the option-only probabilities, and the mass on options."""
        enc, _, y = self.data[split]
        lps = []
        for i in range(0, len(y), self.chunk):
            # keep only the K option columns of each [chunk, V] block: the full matrix for 1,779 rows is 1.8 GB
            full = self.s.last_logp(enc, torch.arange(i, min(i + self.chunk, len(y)), device=self.device))
            lps.append(full[:, self.s.word_ids].clone()); del full
        lp = torch.cat(lps)
        p = (lp - lp.logsumexp(-1, keepdim=True)).exp()
        return lp, p, lp.exp().sum(-1)

    @torch.no_grad()
    def evaluate(self, splits: Sequence[str] = ("val", "test"), forgetting: bool = True) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for split in splits:
            if split not in self.data:
                continue
            _, T, y = self.data[split]
            lp, p, mass = self.option_probs(split)
            ent = (T * T.clamp_min(1e-12).log()).sum(-1)
            k = split + "_"
            m[k + "gap"] = (ent - (T * lp).sum(-1)).mean().item()
            m[k + "kl"] = (ent - (T * p.clamp_min(1e-12).log()).sum(-1)).mean().item()
            m[k + "agree"] = (p.argmax(-1) == T.argmax(-1)).float().mean().item()
            m[k + "acc"] = (p.argmax(-1) == y).float().mean().item()
            m[k + "teacher_acc"] = (T.argmax(-1) == y).float().mean().item()
            m[k + "mass"] = mass.mean().item()
            m[k + "ece"] = ece(p, y)
            m[k + "selective_acc80"] = selective_accuracy(p, y)["accuracy"]
            if split in self.post:
                m[k + "post_kl"] = self.post_kl_value(split)
        if forgetting:
            m.update(self.forgetting())
        return m

    @torch.no_grad()
    def forgetting(self) -> Dict[str, float]:
        lp = self.s.text_logp(self.holdout)
        kl = (self.holdout_lp.exp() * (self.holdout_lp - lp)).sum(-1).mean().item()
        t = self.holdout_tokens; ok = t >= 0
        nll_new = -lp[ok].gather(1, t[ok][:, None]).mean().item()
        nll_old = -self.holdout_lp[ok].gather(1, t[ok][:, None]).mean().item()
        c = self.s.last_logp(self.control)[:, self.s.city_ids]
        cap_acc = (c.argmax(-1) == torch.arange(len(CAPITALS), device=self.device)).float().mean().item()
        return {"holdout_kl": kl, "holdout_ppl": math.exp(nll_new), "holdout_ppl_before": math.exp(nll_old), "capitals_acc": cap_acc}

    # --- what the direct edit reads (forward passes only)
    def _scan(self, layer: int, enc, take) -> None:
        """Run prompts up to `layer`'s down_proj and hand its input [b, L, I] to take(x, attn, last)."""
        mod = down_proj(self.s.model, layer)
        ids, attn, last = enc
        box = []

        def hook(m, inp, out):
            box.append(inp[0]); raise _Stop()

        h = mod.register_forward_hook(hook)
        try:
            for i in range(0, len(ids), self.chunk):
                sl = slice(i, i + self.chunk)
                L = int(last[sl].max().item()) + 1
                box.clear()
                try:
                    self.s.hidden(ids[sl, :L], attn[sl, :L])
                except _Stop:
                    pass
                take(box[0].float(), attn[sl, :L], last[sl])
        finally:
            h.remove()

    @torch.no_grad()
    def keys_and_cov(self, layer: int):
        """K: the down_proj input at the answer position of every training row [n, I]. C: the mean of k k^T over
        everything the edit should leave alone: the guard text, every other token of the training prompts, and the
        position after the answer [I, I]. Both sit upstream of the edit, so they are read once per layer (one pass
        over the training prompts collects both)."""
        if layer in self._kc:
            return self._kc[layer]
        Ks = []
        I = down_proj(self.s.model, layer).weight.shape[1]
        C = torch.zeros(I, I, device=self.device)
        n = [0]

        def acc(X):
            C.addmm_(X.T, X); n[0] += len(X)

        def take_train(x, a, last):
            r = torch.arange(len(x), device=x.device)
            Ks.append(x[r, last])
            m = a.bool().clone(); m[r, last] = False
            acc(x[m])

        self._scan(layer, self.data["train"][0], take_train)
        self._scan(layer, self.guard, lambda x, a, last: acc(x[a.bool()]))
        self._scan(layer, self.post["train"][0], lambda x, a, last: acc(x[torch.arange(len(x), device=x.device), last]))
        self._kc[layer] = (torch.cat(Ks), C / max(n[0], 1), n[0])
        return self._kc[layer]

    def kc_subset(self, parent: "Bench", n_rows: int) -> None:
        """Share a parent bench's keys/covariance when this bench's training rows are the parent's first n_rows."""
        for layer, (K_, C, n) in parent._kc.items():
            self._kc[layer] = (K_[:n_rows], C, n)

    @torch.no_grad()
    def final_state(self, split: str):
        """Residual entering the final norm at the answer position ([n, d] fp32) and the student's full-vocab
        log-probabilities of the option words ([n, K])."""
        enc, _, _ = self.data[split]
        ids, attn, last = enc
        box = []
        h = self.s.base.norm.register_forward_hook(lambda m, inp, out: box.append(inp[0]))
        H, LP = [], []
        try:
            for i in range(0, len(ids), self.chunk):
                sl = slice(i, i + self.chunk)
                L = int(last[sl].max().item()) + 1
                box.clear()
                hs = self.s.hidden(ids[sl, :L], attn[sl, :L])
                r = torch.arange(len(hs), device=self.device)
                H.append(box[0][r, last[sl]].float())
                LP.append(self.s.head(hs[r, last[sl]]).float().log_softmax(-1)[:, self.s.word_ids])
        finally:
            h.remove()
        return torch.cat(H), torch.cat(LP)

    def readout_jacobian(self, H: torch.Tensor) -> torch.Tensor:
        """d(option-word logits)/d(final residual) at each row: final norm and K rows of the unembedding, linearised
        at H. [n, K, d]. A derivative of the readout only, never of the layers."""
        if self._norm32 is None:
            self._norm32 = copy.deepcopy(self.s.base.norm).float()
        W = self.s.head.weight[self.s.word_ids].float()
        with torch.enable_grad():
            Hg = H.detach().clone().requires_grad_(True)
            f = self._norm32(Hg) @ W.T
            J = torch.stack([torch.autograd.grad(f[:, o].sum(), Hg, retain_graph=True)[0] for o in range(f.shape[1])], 1)
        return J.detach()

    @torch.no_grad()
    def residual_targets(self, clip: float = 6.0, ridge: float = 1e-3):
        """dh_i for every training row: the smallest change of the final residual that moves the option-word logits
        by d_i = log(teacher) - log(student), clipped to +-clip. Also returns the training gap before the round."""
        H, lp = self.final_state("train")
        _, T, _ = self.data["train"]
        ent = (T * T.clamp_min(1e-12).log()).sum(-1)
        gap = (ent - (T * lp).sum(-1)).mean().item()
        d = (T.clamp_min(1e-12).log() - lp).clamp(-clip, clip)
        J = self.readout_jacobian(H)
        JJ = J @ J.transpose(1, 2)
        k = JJ.shape[-1]
        lam = ridge * JJ.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)
        JJ = JJ + lam[:, None, None] * torch.eye(k, device=self.device)
        return (J.transpose(1, 2) @ torch.linalg.solve(JJ, d[..., None])).squeeze(-1), gap

    @torch.no_grad()
    def val_and_guard(self):
        """The kept-back rows' gap and the disruption a dose is judged by: guard KL plus post-answer KL on val."""
        return self.gap_value("val"), self.guard_kl() + self.post_kl_value("val")

    # --- detached: plain generation
    def generation_test(self, rows: List[dict], new_tokens: int = 4) -> Dict:
        texts = self.s.generate([prompt(r["text"], self.variant) for r in rows], new_tokens)
        got = [parse_word(t, self.variant) for t in texts]
        gold = [r["gold"] for r in rows]
        teach = [int(torch.tensor(r["teacher"]).argmax()) for r in rows]
        return {"pred": got, "acc": sum(g == y for g, y in zip(got, gold)) / len(rows),
                "agree": sum(g == t for g, t in zip(got, teach)) / len(rows),
                "answered": sum(g is not None for g in got) / len(rows), "texts": texts}


class DirectEditor:
    """Closed-form edits of `width` columns of one down_proj, round after round."""

    def __init__(self, bench: Bench, layer: int, width: int, lam: float, cols: Optional[Sequence[int]] = None):
        self.b = bench
        self.layer, self.lam = layer, lam
        t = time.perf_counter()
        K_, C, n_tok = bench.keys_and_cov(layer)
        if cols is None:
            # the columns most active at the answer position relative to how active they are elsewhere
            score = (K_ ** 2).mean(0) / (C.diagonal() + 1e-6)
            cols = score.topk(min(width, K_.shape[1])).indices.sort().values.tolist()
        self.cols = list(cols)
        c = torch.as_tensor(self.cols, device=bench.device)
        self.Ks = K_[:, c]
        n = len(K_)
        A = self.Ks.T @ self.Ks + lam * n * C[c][:, c]
        A = A + 1e-6 * A.diagonal().mean() * torch.eye(len(c), device=bench.device)
        self.A = A
        self.read_s = time.perf_counter() - t
        self.edit = ColumnEdit(bench.s.model, layer, self.cols)

    def solve(self, R: torch.Tensor):
        """dW_S = R^T K_S A^-1, timed: this is the edit itself."""
        sync(self.b.device); t = time.perf_counter()
        dW = torch.linalg.solve(self.A, self.Ks.T @ R).T.contiguous()
        sync(self.b.device)
        return dW, (time.perf_counter() - t) * 1000

    @torch.no_grad()
    def round(self, doses: Sequence[float], guard_weight: float, guard_cap: float = float("inf")) -> Dict:
        R, train_gap = self.b.residual_targets()
        dW, solve_ms = self.solve(R)
        base = self.edit.D.clone()
        g0, k0 = self.b.val_and_guard()
        best_a, best, best_k = 0.0, g0 + guard_weight * k0, k0
        refused = []
        for a in doses:
            self.edit.D.copy_(base + a * dW)
            g, k = self.b.val_and_guard()
            if k > guard_cap:
                refused.append(a); continue
            if g + guard_weight * k < best:
                best_a, best, best_k = a, g + guard_weight * k, k
        sync(self.b.device); t = time.perf_counter()
        self.edit.D.copy_(base + best_a * dW)
        sync(self.b.device)
        return {"solve_ms": solve_ms, "write_ms": (time.perf_counter() - t) * 1000, "dose": best_a, "refused": refused,
                "objective": best, "guard_kl": best_k, "train_gap_before": train_gap,
                "dW_norm": dW.norm().item() * best_a, "D_norm": self.edit.D.norm().item()}

    def run(self, rounds: int, doses: Sequence[float], guard_weight: float, guard_cap: float = float("inf"),
            eval_splits=("val", "test"), log=None, forgetting: bool = True) -> Dict:
        """Rounds until no allowed dose helps, then back to the best round on the kept-back rows (agreement with the
        teacher first, gap second)."""
        hist = [{"round": 0, "guard_kl": 0.0, **self.b.evaluate(eval_splits, forgetting)}]
        snaps = [self.edit.D.clone()]
        for r in range(1, rounds + 1):
            info = self.round(doses, guard_weight, guard_cap)
            hist.append({"round": r, **info, **self.b.evaluate(eval_splits, forgetting)})
            snaps.append(self.edit.D.clone())
            if log:
                log(hist[-1])
            if info["dose"] == 0.0:
                break
        best = len(hist) - 1
        if "val" in eval_splits:
            best = max(range(len(hist)), key=lambda i: (hist[i]["val_agree"], -hist[i]["val_gap"]))
        self.edit.D.copy_(snaps[best])
        return {"hist": hist, "best_round": best}


def smoothed_one_hot(gold: int, eps: float = 0.03) -> List[float]:
    v = [eps / (K - 1)] * K
    v[gold] = 1.0 - eps
    return v


def random_like(D: torch.Tensor, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(D.shape, generator=g).to(D.device)
    return R * (D.norm() / R.norm())

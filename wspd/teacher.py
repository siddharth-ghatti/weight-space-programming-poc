"""The Jev-like teacher: calibrated decision heads on the frozen reader.

Features are the reader's residual stream at the answer position (and the mean over sentence tokens) at a few
layers, read once with hooks. The head is a multinomial logistic regression chosen by 5-fold cross-validation on
the training rows only, followed by one temperature fitted on out-of-fold probabilities. It answers a typed
choice question with calibrated probabilities over the 8 options, a confidence 1 - H/log K and an abstain flag,
and never generates text: the same construction as a vegaml/Laya policy head, under our control."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from wspd.common import K, ece, selective_accuracy
from wspd.reader import Student

DEFAULT_LAYERS = (12, 16, 20)
FEATURE_SETS = {
    "final_last": ["final_last"],
    "L16_last": ["L16_last"],
    "L20_last": ["L20_last"],
    "final_last+L16_mean": ["final_last", "L16_mean"],
    "L12_last+L16_last+L20_last+final_last": ["L12_last", "L16_last", "L20_last", "final_last"],
    "final_last+final_mean": ["final_last", "final_mean"],
    "all": ["L12_last", "L16_last", "L20_last", "final_last", "L16_mean", "final_mean"],
}
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


@torch.no_grad()
def read_features(student: Student, prompts: Sequence[str], layers: Sequence[int] = DEFAULT_LAYERS, chunk: int = 16,
                  sent_spans: Sequence[tuple[int, int]] | None = None) -> dict[str, np.ndarray]:
    """Residual stream at the answer position after layers `layers` and after the final norm, plus the mean over
    all real tokens at layer 16 and after the final norm. Returns {name: [n, d] float32}."""
    base = student.base
    boxes: dict[int, torch.Tensor] = {}
    handles = [base.layers[l].register_forward_hook(lambda m, i, o, l=l: boxes.__setitem__(l, (o[0] if isinstance(o, tuple) else o)))
               for l in layers]
    feats: dict[str, list[np.ndarray]] = {}
    try:
        for i in range(0, len(prompts), chunk):
            ids, attn, last = student.encode(prompts[i:i + chunk])
            L = int(last.max().item()) + 1
            ids, attn = ids[:, :L], attn[:, :L]
            boxes.clear()
            hfin = student.hidden(ids, attn)
            r = torch.arange(len(ids), device=student.device)
            m = attn.unsqueeze(-1).to(hfin.dtype)
            def put(name, x):
                feats.setdefault(name, []).append(x.float().cpu().numpy())
            put("final_last", hfin[r, last])
            put("final_mean", (hfin * m).sum(1) / m.sum(1))
            for l in layers:
                h = boxes[l]
                put(f"L{l}_last", h[r, last])
                if l == 16:
                    put("L16_mean", (h * m).sum(1) / m.sum(1))
    finally:
        for h in handles:
            h.remove()
    return {k: np.concatenate(v) for k, v in feats.items()}


def _stack(feats: dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.concatenate([feats[n] for n in names], axis=1)


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One scalar temperature minimizing NLL of softmax(logits / T) on out-of-fold logits (grid on log T)."""
    best, best_t = None, 1.0
    for lt in np.linspace(math.log(0.2), math.log(5.0), 81):
        t = math.exp(lt)
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(y)), y].mean()
        if best is None or nll < best:
            best, best_t = nll, t
    return best_t


class Teacher:
    def __init__(self, feature_names: Sequence[str], scaler_mean, scaler_scale, coef, intercept, temperature: float,
                 abstain_threshold: float, report: dict | None = None):
        self.feature_names = list(feature_names)
        self.mean = np.asarray(scaler_mean, dtype=np.float64); self.scale = np.asarray(scaler_scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64); self.intercept = np.asarray(intercept, dtype=np.float64)
        self.temperature = float(temperature); self.abstain_threshold = float(abstain_threshold)
        self.report = report or {}

    def logits(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        x = (_stack(feats, self.feature_names) - self.mean) / self.scale
        return x @ self.coef.T + self.intercept

    def probs(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        z = self.logits(feats) / self.temperature
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); return p / p.sum(1, keepdims=True)

    def decide(self, feats: dict[str, np.ndarray]) -> list[dict]:
        """Jev-style typed answer per row: choice, probabilities, confidence, abstain."""
        from wsp.labels import LABELS
        P = self.probs(feats)
        out = []
        for p in P:
            conf = 1 - float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(K))
            out.append({"choice": LABELS[int(p.argmax())], "probabilities": {LABELS[i]: float(p[i]) for i in range(K)},
                        "confidence": conf, "abstain": conf < self.abstain_threshold})
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "head.npz", mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept)
        (path / "teacher.json").write_text(json.dumps({"feature_names": self.feature_names, "temperature": self.temperature,
                                                       "abstain_threshold": self.abstain_threshold, "report": self.report}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Teacher":
        j = json.loads((path / "teacher.json").read_text()); h = np.load(path / "head.npz")
        return cls(j["feature_names"], h["mean"], h["scale"], h["coef"], h["intercept"], j["temperature"],
                   j["abstain_threshold"], j.get("report"))


def fit_teacher(feats: dict[str, np.ndarray], y: np.ndarray, *, folds: int = 5, seed: int = 0,
                feature_sets: dict[str, list[str]] = FEATURE_SETS, c_grid: Sequence[float] = C_GRID,
                coverage: float = 0.8, log=print) -> Teacher:
    """Choose the feature set and C by out-of-fold accuracy, refit on everything, calibrate on the OOF logits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    results = []
    for fname, names in feature_sets.items():
        X = _stack(feats, names)
        for C in c_grid:
            oof = np.zeros((len(y), K))
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
                oof[te] = clf.decision_function(sc.transform(X[te]))
            acc = float((oof.argmax(1) == y).mean())
            results.append({"features": fname, "C": C, "oof_accuracy": acc, "oof_logits": oof})
            log(f"  {fname:42s} C={C:<5} oof acc {acc:.4f}")
    best = max(results, key=lambda r: r["oof_accuracy"])
    names = feature_sets[best["features"]]
    X = _stack(feats, names)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=best["C"], max_iter=3000).fit(sc.transform(X), y)
    T = _fit_temperature(best["oof_logits"], y)
    z = best["oof_logits"] / T; z = z - z.max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
    Pt, yt = torch.tensor(P), torch.tensor(y)
    sel = selective_accuracy(Pt, yt, coverage)
    report = {"chosen": {"features": best["features"], "C": best["C"]}, "oof_accuracy": best["oof_accuracy"],
              "oof_ece_after_calibration": ece(Pt, yt), "temperature": T, "oof_selective": sel,
              "grid": [{k: v for k, v in r.items() if k != "oof_logits"} for r in results], "n_train": int(len(y))}
    return Teacher(names, sc.mean_, sc.scale_, clf.coef_, clf.intercept_, T, sel["threshold"], report)


def score_teacher(teacher: Teacher, feats: dict[str, np.ndarray], y: np.ndarray) -> dict:
    from wsp.eval.evaluate import full_metrics
    P = teacher.probs(feats)
    pred = P.argmax(1)
    m = full_metrics(y.tolist(), pred.tolist())
    Pt, yt = torch.tensor(P), torch.tensor(y)
    m["ece"] = ece(Pt, yt); m["selective"] = selective_accuracy(Pt, yt)
    m["abstain_rate"] = float(np.mean([d["abstain"] for d in teacher.decide(feats)]))
    return m

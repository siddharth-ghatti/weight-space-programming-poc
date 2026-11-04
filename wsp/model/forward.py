"""Harness-side forward-pass executors. These run OUTSIDE the sandbox with torch.

Two kinds of pass, each costing one budget unit:
  predict: run base and/or edited classifier on practice items -> labels, logits, metrics
  read:    teacher-forced read of hidden states / pooled / logits / attention -> .npy files
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from wsp.labels import ID2LABEL, LABEL2ID, LABELS, NUM_LABELS
from wsp.model import weights as W

MAX_LEN = 128
BATCH = 64
MAX_PREDICT = 512
MAX_READ = 1024
MAX_ATTN_TEXTS = 64


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bert_config() -> BertConfig:
    cfg = BertConfig.from_pretrained(str(W.BASE_DIR), attn_implementation="eager")
    cfg.num_labels = NUM_LABELS
    cfg.id2label = {i: l for i, l in ID2LABEL.items()}
    cfg.label2id = dict(LABEL2ID)
    return cfg


def build_model(weights: dict[str, np.ndarray], device: torch.device | None = None) -> BertForSequenceClassification:
    device = device or pick_device()
    cfg = bert_config()
    model = BertForSequenceClassification(cfg)
    state = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in W.model_state(weights).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    return model


_tok: BertTokenizerFast | None = None


def tokenizer() -> BertTokenizerFast:
    global _tok
    if _tok is None:
        _tok = BertTokenizerFast.from_pretrained(str(W.BASE_DIR))
    return _tok


@torch.no_grad()
def run_batches(model, texts: list[str], want_hidden=False, want_attn=False):
    tok = tokenizer()
    dev = next(model.parameters()).device
    logits, pooled, hidden, attn, masks = [], [], [], [], []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        out = model(**enc, output_hidden_states=want_hidden, output_attentions=want_attn)
        logits.append(out.logits.float().cpu().numpy())
        if want_hidden:
            hidden.append([h.float().cpu().numpy() for h in out.hidden_states])
            # pooled = tanh(W_p · h_cls + b_p) — recompute explicitly
            h_cls = out.hidden_states[-1][:, 0]
            pooled.append(torch.tanh(model.bert.pooler.dense(h_cls)).float().cpu().numpy())
            masks.append(enc["attention_mask"].cpu().numpy())
        if want_attn:
            attn.append([a.float().cpu().numpy() for a in out.attentions])
    return {
        "logits": np.concatenate(logits),
        "hidden": hidden,
        "pooled": np.concatenate(pooled) if pooled else None,
        "attn": attn,
        "masks": masks,
    }


def metrics(gold: list[int], pred: list[int]) -> dict:
    gold_a, pred_a = np.array(gold), np.array(pred)
    acc = float((gold_a == pred_a).mean()) if len(gold) else 0.0
    cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    for g, p in zip(gold_a, pred_a):
        cm[g, p] += 1
    f1s = []
    for c in range(NUM_LABELS):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
    return {
        "n": len(gold),
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": {LABELS[c]: float(f1s[c]) for c in range(NUM_LABELS)},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": LABELS,
    }


# ---------------------------------------------------------------- executors

class ForwardContext:
    """Holds base weights, practice data, and builds models on demand (cached per edit hash)."""

    def __init__(self, base: dict[str, np.ndarray], practice: list[dict], work_dir: Path):
        self.base = base
        self.practice = practice
        self.by_id = {p["id"]: p for p in practice}
        self.work_dir = work_dir
        self.device = pick_device()
        self._cache: dict[str, BertForSequenceClassification] = {}

    def current_edits(self) -> dict[str, np.ndarray]:
        return W.load_edits(self.work_dir / "edits")

    def model_for(self, which: str) -> BertForSequenceClassification:
        if which == "base":
            key = "base"
            weights = self.base
        elif which == "edited":
            edits = self.current_edits()
            key = "edited:" + ",".join(f"{k}:{W.sha256(v)[:12]}" for k, v in sorted(edits.items()))
            weights = W.apply_edits(self.base, edits)
        else:
            raise ValueError(f"model must be 'base' or 'edited', got {which!r}")
        if key not in self._cache:
            self._cache.clear()  # keep memory bounded: one model at a time besides base
            self._cache[key] = build_model(weights, self.device)
        return self._cache[key]

    def select_items(self, ids: list[str] | None, n: int, seed: int) -> list[dict]:
        if ids:
            unknown = [i for i in ids if i not in self.by_id]
            if unknown:
                raise ValueError(f"unknown practice ids: {unknown[:10]}")
            return [self.by_id[i] for i in ids]
        rng = random.Random(seed)
        return rng.sample(self.practice, min(n, len(self.practice)))


def predict_pass(ctx: ForwardContext, req: dict, pass_dir: Path) -> dict:
    which = req.get("model", "edited")
    n = int(req.get("n", 256))
    if n < 1 or n > MAX_PREDICT:
        raise ValueError(f"n must be in [1, {MAX_PREDICT}]")
    ids = req.get("ids") or None
    if ids and len(ids) > MAX_PREDICT:
        raise ValueError(f"at most {MAX_PREDICT} ids per pass")
    seed = int(req.get("seed", 0))
    models = ["base", "edited"] if which == "both" else [which]
    if which == "both" and not ids:
        n = min(n, 256)
    items = ctx.select_items(ids, n, seed)
    texts = [it["text"] for it in items]
    gold = [LABEL2ID[it["label"]] for it in items]
    pass_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"kind": "predict", "n": len(items), "models": {}}
    per_item = {it["id"]: {"id": it["id"], "text": it["text"], "gold": it["label"]} for it in items}
    for m in models:
        model = ctx.model_for(m)
        out = run_batches(model, texts)
        pred = out["logits"].argmax(-1).tolist()
        summary["models"][m] = metrics(gold, pred)
        for it, p, lg in zip(items, pred, out["logits"]):
            per_item[it["id"]][f"pred_{m}"] = ID2LABEL[p]
            per_item[it["id"]][f"logits_{m}"] = [round(float(x), 4) for x in lg]
    (pass_dir / "predictions.json").write_text(json.dumps(list(per_item.values()), indent=1))
    summary["files"] = {"predictions": str(pass_dir / "predictions.json")}
    summary["note"] = (
        "predictions.json has per-item gold, predicted label(s) and logits. "
        "Confusion matrix rows = gold, cols = predicted, in label order."
    )
    return summary


READ_OUTPUTS = {"cls_hidden", "mean_hidden", "pooled", "logits", "attention", "token_hidden"}


def read_pass(ctx: ForwardContext, req: dict, pass_dir: Path) -> dict:
    which = req.get("model", "base")
    if which not in ("base", "edited"):
        raise ValueError("model must be 'base' or 'edited' for a read pass")
    outputs = set(req.get("outputs") or ["cls_hidden"])
    bad = outputs - READ_OUTPUTS
    if bad:
        raise ValueError(f"unknown outputs {sorted(bad)}; allowed: {sorted(READ_OUTPUTS)}")
    texts = req.get("texts") or None
    ids = req.get("ids") or None
    if texts and ids:
        raise ValueError("give either texts or ids, not both")
    if ids:
        items = ctx.select_items(ids, len(ids), 0)
        texts = [it["text"] for it in items]
        labels = [it["label"] for it in items]
    elif texts:
        labels = None
    else:
        raise ValueError("a read pass needs texts or ids")
    if len(texts) > MAX_READ:
        raise ValueError(f"at most {MAX_READ} texts per read pass")
    if "attention" in outputs and len(texts) > MAX_ATTN_TEXTS:
        raise ValueError(f"attention output is limited to {MAX_ATTN_TEXTS} texts per pass")
    if "token_hidden" in outputs and len(texts) > MAX_ATTN_TEXTS:
        raise ValueError(f"token_hidden output is limited to {MAX_ATTN_TEXTS} texts per pass")
    layers = req.get("layers")
    layers = list(range(13)) if not layers else [int(l) for l in layers]
    if any(l < 0 or l > 12 for l in layers):
        raise ValueError("layers must be in 0..12 (0 = embedding output, 12 = last layer)")

    model = ctx.model_for(which)
    want_hidden = bool(outputs & {"cls_hidden", "mean_hidden", "pooled", "token_hidden"})
    want_attn = "attention" in outputs
    out = run_batches(model, texts, want_hidden=want_hidden, want_attn=want_attn)
    pass_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}

    def save(name: str, arr: np.ndarray):
        p = pass_dir / f"{name}.npy"
        np.save(p, arr.astype(np.float32))
        files[name] = str(p)
        shapes[name] = list(arr.shape)

    meta = {"texts": texts, "labels": labels, "model": which, "layers": layers}
    (pass_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    files["meta"] = str(pass_dir / "meta.json")

    if "logits" in outputs:
        save("logits", out["logits"])
    if want_hidden:
        # out["hidden"] is list over batches of list over 13 layers of [b, T, 768]
        masks = out["masks"]
        for l in layers:
            cls_parts, mean_parts = [], []
            for hb, mb in zip(out["hidden"], masks):
                h = hb[l]
                cls_parts.append(h[:, 0])
                m = mb[:, :, None].astype(np.float32)
                mean_parts.append((h * m).sum(1) / np.maximum(m.sum(1), 1))
            if "cls_hidden" in outputs:
                save(f"cls_hidden_L{l}", np.concatenate(cls_parts))
            if "mean_hidden" in outputs:
                save(f"mean_hidden_L{l}", np.concatenate(mean_parts))
        if "pooled" in outputs:
            save("pooled", out["pooled"])
        if "token_hidden" in outputs:
            # pad to common length T_max across batches
            T = max(hb[0].shape[1] for hb in out["hidden"])
            for l in layers:
                parts = []
                for hb in out["hidden"]:
                    h = hb[l]
                    parts.append(np.pad(h, ((0, 0), (0, T - h.shape[1]), (0, 0))))
                save(f"token_hidden_L{l}", np.concatenate(parts))
            mparts = [np.pad(mb, ((0, 0), (0, T - mb.shape[1]))) for mb in masks]
            save("attention_mask", np.concatenate(mparts))
            tok = tokenizer()
            toks = [tok.convert_ids_to_tokens(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts]
            (pass_dir / "tokens.json").write_text(json.dumps(toks))
            files["tokens"] = str(pass_dir / "tokens.json")
    if want_attn:
        T = max(ab[0].shape[-1] for ab in out["attn"])
        for l in layers:
            if l == 0:
                continue  # attention is per encoder layer 1..12
            parts = []
            for ab in out["attn"]:
                a = ab[l - 1]  # [b, heads, T, T]
                parts.append(np.pad(a, ((0, 0), (0, 0), (0, T - a.shape[2]), (0, T - a.shape[3]))))
            save(f"attention_L{l}", np.concatenate(parts))
        if "tokens" not in files:
            tok = tokenizer()
            toks = [tok.convert_ids_to_tokens(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts]
            (pass_dir / "tokens.json").write_text(json.dumps(toks))
            files["tokens"] = str(pass_dir / "tokens.json")

    return {
        "kind": "read",
        "model": which,
        "n_texts": len(texts),
        "layers": layers,
        "files": files,
        "shapes": shapes,
        "note": (
            "Arrays are float32 .npy in the pass folder; row i corresponds to texts[i] in meta.json. "
            "Hidden layer index 0 is the embedding output, 12 is the final encoder layer. "
            "Attention arrays are [n, heads, T, T] for encoder layers 1..12."
        ),
    }


def run_pass(ctx: ForwardContext, req: dict, pass_dir: Path) -> dict:
    kind = req.get("kind")
    if kind == "predict":
        return predict_pass(ctx, req, pass_dir)
    if kind == "read":
        return read_pass(ctx, req, pass_dir)
    raise ValueError("kind must be 'predict' or 'read'")

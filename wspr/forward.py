"""Harness-side forward passes (outside the sandbox, with torch) for the reinsurance pruning harness.

`base` = the unpruned start checkpoint. `edited` = start + saved edits, pruned by the editor's current
structure.json. Inputs are paragraphs, so MAX_LEN is the model's full 512-token window.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from wspr.arch import Arch
from wspr.bundle import apply_edits, load_edits
from wspr.start import tensor_sha256
from wspr.structure import Structure, load_structure, param_count, prune_weights, structure_from_weights, structure_hash

MAX_LEN = 512
BATCH = 16
MAX_PREDICT = 1024
MAX_READ = 1024
MAX_ATTN_TEXTS = 32


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def model_config(arch: Arch, num_layers: int | None = None, intermediate_size: int | None = None):
    cfg = AutoConfig.from_pretrained(str(arch.model_dir), attn_implementation="eager")
    cfg.num_hidden_layers = int(num_layers if num_layers is not None else arch.num_layers)
    cfg.intermediate_size = int(intermediate_size if intermediate_size is not None else arch.ffn)
    cfg.num_labels = arch.num_labels
    cfg.id2label = dict(arch.id2label)
    cfg.label2id = dict(arch.label2id)
    return cfg


def build_model(weights: dict[str, np.ndarray], arch: Arch, device: torch.device | None = None, *,
                structure: Structure | None = None):
    """Build from full-shaped weights (pruned here if a non-full structure is given) or from
    already-pruned weights (structure=None: config inferred from the tensors)."""
    device = device or pick_device()
    if structure is not None and not structure.is_full(arch):
        weights = prune_weights(weights, structure, arch)
    k, m = structure_from_weights(weights, arch)
    model = AutoModelForSequenceClassification.from_config(model_config(arch, k, m))
    state = {kk: torch.from_numpy(np.ascontiguousarray(v)) for kk, v in weights.items() if not arch.is_buffer(kk)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [x for x in missing if "position_ids" not in x]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    return model


_tok_cache: dict[str, object] = {}


def tokenizer(arch: Arch):
    key = str(arch.model_dir)
    if key not in _tok_cache:
        _tok_cache[key] = AutoTokenizer.from_pretrained(key)
    return _tok_cache[key]


def head_pooled(model, h_cls: torch.Tensor) -> torch.Tensor:
    """The classification head's intermediate: tanh(dense(h[<s>])) for RobertaClassificationHead."""
    return torch.tanh(model.classifier.dense(h_cls))


@torch.no_grad()
def run_batches(model, tok, texts: list[str], want_hidden=False, want_attn=False, max_len: int = MAX_LEN, batch: int = BATCH):
    dev = next(model.parameters()).device
    logits, pooled, hidden, attn, masks = [], [], [], [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        out = model(**enc, output_hidden_states=want_hidden, output_attentions=want_attn)
        logits.append(out.logits.float().cpu().numpy())
        if want_hidden:
            hidden.append([h.float().cpu().numpy() for h in out.hidden_states])
            pooled.append(head_pooled(model, out.hidden_states[-1][:, 0]).float().cpu().numpy())
            masks.append(enc["attention_mask"].cpu().numpy())
        if want_attn:
            attn.append([a.float().cpu().numpy() for a in out.attentions])
    return {"logits": np.concatenate(logits), "hidden": hidden, "pooled": np.concatenate(pooled) if pooled else None,
            "attn": attn, "masks": masks}


def metrics(gold: list[int], pred: list[int], labels: tuple[str, ...]) -> dict:
    n_lab = len(labels)
    gold_a, pred_a = np.array(gold), np.array(pred)
    acc = float((gold_a == pred_a).mean()) if len(gold) else 0.0
    cm = np.zeros((n_lab, n_lab), dtype=int)
    for g, p in zip(gold_a, pred_a):
        cm[g, p] += 1
    f1s, recalls = [], []
    for c in range(n_lab):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
        recalls.append(r_)
    return {"n": len(gold), "accuracy": acc, "macro_f1": float(np.mean(f1s)),
            "per_class_f1": {labels[c]: float(f1s[c]) for c in range(n_lab)},
            "per_class_recall": {labels[c]: float(recalls[c]) for c in range(n_lab)},
            "confusion_matrix": cm.tolist(), "confusion_labels": list(labels)}


class PrunedContext:
    def __init__(self, arch: Arch, start: dict[str, np.ndarray], practice: list[dict], work_dir: Path):
        self.arch = arch
        self.start = start
        self.practice = practice
        self.by_id = {p["id"]: p for p in practice}
        self.work_dir = Path(work_dir)
        self.device = pick_device()
        self.tok = tokenizer(arch)
        self._cache: dict[str, object] = {}

    def current_edits(self) -> dict[str, np.ndarray]:
        return load_edits(self.work_dir / "edits")

    def current_structure(self) -> Structure:
        return load_structure(self.work_dir, self.arch)

    def num_layers(self, which: str) -> int:
        return self.arch.num_layers if which == "base" else self.current_structure().k

    def model_for(self, which: str):
        if which == "base":
            key, weights, structure = "base", self.start, None
        elif which == "edited":
            structure = self.current_structure()
            edits = self.current_edits()
            key = "edited:" + structure_hash(structure, self.arch) + ":" + ",".join(f"{k}:{tensor_sha256(v)[:12]}" for k, v in sorted(edits.items()))
            weights = apply_edits(self.start, edits, self.arch)
        else:
            raise ValueError(f"model must be 'base' or 'edited', got {which!r}")
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = build_model(weights, self.arch, self.device, structure=structure)
        return self._cache[key]

    def select_items(self, ids: list[str] | None, n: int, seed: int) -> list[dict]:
        if ids:
            unknown = [i for i in ids if i not in self.by_id]
            if unknown:
                raise ValueError(f"unknown practice ids: {unknown[:10]}")
            return [self.by_id[i] for i in ids]
        return random.Random(seed).sample(self.practice, min(n, len(self.practice)))


def _structure_summary(s: Structure, arch: Arch) -> dict:
    pc = param_count(s, arch)
    return {"layers": list(s.layers), "intermediate_size": s.m, "params": pc["total"],
            "reduction_vs_full": round(pc["reduction_vs_full"], 4),
            "non_embedding_reduction_vs_full": round(pc["non_embedding_reduction_vs_full"], 4)}


def predict_pass(ctx: PrunedContext, req: dict, pass_dir: Path) -> dict:
    arch = ctx.arch
    which = req.get("model", "edited")
    if which not in ("base", "edited", "both"):
        raise ValueError("model must be 'base', 'edited' or 'both'")
    n = int(req.get("n", 256))
    if n < 1 or n > MAX_PREDICT:
        raise ValueError(f"n must be in [1, {MAX_PREDICT}]")
    ids = req.get("ids") or None
    if ids and len(ids) > MAX_PREDICT:
        raise ValueError(f"at most {MAX_PREDICT} ids per pass")
    if req.get("texts"):
        raise ValueError("free texts are not allowed; use practice ids")
    seed = int(req.get("seed", 0))
    models = ["base", "edited"] if which == "both" else [which]
    if "edited" in models:
        structure = ctx.current_structure()
    items = ctx.select_items(ids, n, seed)
    texts = [it["text"] for it in items]
    gold = [arch.label2id[it["label"]] for it in items]
    pass_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"kind": "predict", "n": len(items), "models": {}}
    per_item = {it["id"]: {"id": it["id"], "gold": it["label"]} for it in items}
    for m in models:
        model = ctx.model_for(m)
        out = run_batches(model, ctx.tok, texts)
        pred = out["logits"].argmax(-1).tolist()
        summary["models"][m] = metrics(gold, pred, arch.labels)
        for it, p, lg in zip(items, pred, out["logits"]):
            per_item[it["id"]][f"pred_{m}"] = arch.labels[p]
            per_item[it["id"]][f"logits_{m}"] = [round(float(x), 4) for x in lg]
    if "edited" in models:
        summary["structure"] = _structure_summary(structure, arch)
    (pass_dir / "predictions.json").write_text(json.dumps(list(per_item.values()), indent=1))
    summary["files"] = {"predictions": str(pass_dir / "predictions.json")}
    summary["note"] = ("predictions.json has per-item id, gold, predicted label(s) and logits (texts are in practice.jsonl). "
                       "Confusion matrix rows = gold, cols = predicted, in label order.")
    return summary


READ_OUTPUTS = {"cls_hidden", "mean_hidden", "pooled", "logits", "attention", "token_hidden"}


def read_pass(ctx: PrunedContext, req: dict, pass_dir: Path) -> dict:
    arch = ctx.arch
    which = req.get("model", "base")
    if which not in ("base", "edited"):
        raise ValueError("model must be 'base' or 'edited' for a read pass")
    outputs = set(req.get("outputs") or ["cls_hidden"])
    bad = outputs - READ_OUTPUTS
    if bad:
        raise ValueError(f"unknown outputs {sorted(bad)}; allowed: {sorted(READ_OUTPUTS)}")
    if req.get("texts"):
        raise ValueError("free texts are not allowed; use practice ids")
    ids = req.get("ids") or None
    if not ids:
        raise ValueError("a read pass needs practice ids")
    items = ctx.select_items(ids, len(ids), 0)
    texts = [it["text"] for it in items]
    labels = [it["label"] for it in items]
    if len(texts) > MAX_READ:
        raise ValueError(f"at most {MAX_READ} ids per read pass")
    if ("attention" in outputs or "token_hidden" in outputs) and len(texts) > MAX_ATTN_TEXTS:
        raise ValueError(f"attention/token_hidden outputs are limited to {MAX_ATTN_TEXTS} ids per pass (paragraph inputs)")
    L = ctx.num_layers(which)
    structure = ctx.current_structure() if which == "edited" else None
    layers = req.get("layers")
    layers = list(range(L + 1)) if not layers else [int(l) for l in layers]
    if any(l < 0 or l > L for l in layers):
        raise ValueError(f"layers must be in 0..{L} for the {which} model (0 = embedding output, {L} = last encoder layer)")

    model = ctx.model_for(which)
    want_hidden = bool(outputs & {"cls_hidden", "mean_hidden", "pooled", "token_hidden"})
    want_attn = "attention" in outputs
    out = run_batches(model, ctx.tok, texts, want_hidden=want_hidden, want_attn=want_attn)
    pass_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}

    def save(name: str, arr: np.ndarray):
        p = pass_dir / f"{name}.npy"
        np.save(p, arr.astype(np.float32))
        files[name] = str(p)
        shapes[name] = list(arr.shape)

    meta = {"ids": [it["id"] for it in items], "labels": labels, "model": which, "layers": layers, "num_layers": L}
    if structure is not None:
        meta["layer_map"] = {str(k): v for k, v in structure.layer_map().items()}
        meta["structure"] = _structure_summary(structure, arch)
    (pass_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    files["meta"] = str(pass_dir / "meta.json")

    def write_tokens():
        if "tokens" not in files:
            toks = [ctx.tok.convert_ids_to_tokens(ctx.tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts]
            (pass_dir / "tokens.json").write_text(json.dumps(toks))
            files["tokens"] = str(pass_dir / "tokens.json")

    if "logits" in outputs:
        save("logits", out["logits"])
    if want_hidden:
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
            T = max(hb[0].shape[1] for hb in out["hidden"])
            for l in layers:
                parts = [np.pad(hb[l], ((0, 0), (0, T - hb[l].shape[1]), (0, 0))) for hb in out["hidden"]]
                save(f"token_hidden_L{l}", np.concatenate(parts))
            save("attention_mask", np.concatenate([np.pad(mb, ((0, 0), (0, T - mb.shape[1]))) for mb in masks]))
            write_tokens()
    if want_attn:
        T = max(ab[0].shape[-1] for ab in out["attn"])
        for l in layers:
            if l == 0:
                continue
            parts = []
            for ab in out["attn"]:
                a = ab[l - 1]
                parts.append(np.pad(a, ((0, 0), (0, 0), (0, T - a.shape[2]), (0, T - a.shape[3]))))
            save(f"attention_L{l}", np.concatenate(parts))
        write_tokens()

    note = (f"Arrays are float32 .npy in the pass folder; row i corresponds to ids[i] in meta.json. "
            f"Hidden layer index 0 is the embedding output, {L} is the last encoder layer of the {which} model. "
            f"cls_hidden is position 0 (the <s> token, which feeds the classification head). "
            f"Attention arrays are [n, {arch.heads} heads, T, T] for encoder layers 1..{L}.")
    if structure is not None:
        note += " meta.json has layer_map: edited-model layer index -> original layer id."
    result = {"kind": "read", "model": which, "n_texts": len(texts), "layers": layers, "num_layers": L,
              "files": files, "shapes": shapes, "note": note}
    if structure is not None:
        result["layer_map"] = meta["layer_map"]
        result["structure"] = meta["structure"]
    return result


def run_pass(ctx: PrunedContext, req: dict, pass_dir: Path) -> dict:
    kind = req.get("kind")
    if kind == "predict":
        return predict_pass(ctx, req, pass_dir)
    if kind == "read":
        return read_pass(ctx, req, pass_dir)
    raise ValueError("kind must be 'predict' or 'read'")

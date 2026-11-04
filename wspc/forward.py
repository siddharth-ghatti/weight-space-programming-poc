"""Harness-side forward passes for the pruning experiment.

`base` = the unpruned 12-layer start model. `edited` = start + saved edits, pruned by the editor's
current structure.json. Reuses the batching, tokenizer and metrics of `wsp.model.forward`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from wsp.labels import ID2LABEL, LABEL2ID, NUM_LABELS
from wsp.model import weights as W
from wsp.model.forward import (BATCH, MAX_ATTN_TEXTS, MAX_LEN, MAX_PREDICT, MAX_READ, metrics, pick_device,
                               run_batches, tokenizer)
from wspc.structure import (FULL_FFN, FULL_LAYERS, Structure, StructureError, load_structure, param_count,
                            prune_weights, structure_from_weights, structure_hash)


def bert_config(num_layers: int = FULL_LAYERS, intermediate_size: int = FULL_FFN) -> BertConfig:
    cfg = BertConfig.from_pretrained(str(W.BASE_DIR), attn_implementation="eager")
    cfg.num_hidden_layers = int(num_layers)
    cfg.intermediate_size = int(intermediate_size)
    cfg.num_labels = NUM_LABELS
    cfg.id2label = {i: l for i, l in ID2LABEL.items()}
    cfg.label2id = dict(LABEL2ID)
    return cfg


def build_model(weights: dict[str, np.ndarray], device: torch.device | None = None, *,
                structure: Structure | None = None) -> BertForSequenceClassification:
    """Build from full-shaped weights (pruning them if a non-full structure is given) or from
    already-pruned weights (structure=None: config inferred from the tensors)."""
    device = device or pick_device()
    if structure is not None and not structure.is_full():
        weights = prune_weights(weights, structure)
    k, m = structure_from_weights(weights)
    model = BertForSequenceClassification(bert_config(k, m))
    state = {kk: torch.from_numpy(np.ascontiguousarray(v)) for kk, v in W.model_state(weights).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [x for x in missing if "position_ids" not in x]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    return model


class PrunedContext:
    def __init__(self, start: dict[str, np.ndarray], practice: list[dict], work_dir: Path):
        self.start = start
        self.practice = practice
        self.by_id = {p["id"]: p for p in practice}
        self.work_dir = Path(work_dir)
        self.device = pick_device()
        self._cache: dict[str, BertForSequenceClassification] = {}

    def current_edits(self) -> dict[str, np.ndarray]:
        return W.load_edits(self.work_dir / "edits")

    def current_structure(self) -> Structure:
        return load_structure(self.work_dir)  # raises StructureError on an invalid file

    def num_layers(self, which: str) -> int:
        return FULL_LAYERS if which == "base" else self.current_structure().k

    def model_for(self, which: str) -> BertForSequenceClassification:
        if which == "base":
            key, weights, structure = "base", self.start, None
        elif which == "edited":
            structure = self.current_structure()
            edits = self.current_edits()
            key = "edited:" + structure_hash(structure) + ":" + ",".join(f"{k}:{W.sha256(v)[:12]}" for k, v in sorted(edits.items()))
            weights = W.apply_edits(self.start, edits)
        else:
            raise ValueError(f"model must be 'base' or 'edited', got {which!r}")
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = build_model(weights, self.device, structure=structure)
        return self._cache[key]

    def select_items(self, ids: list[str] | None, n: int, seed: int) -> list[dict]:
        if ids:
            unknown = [i for i in ids if i not in self.by_id]
            if unknown:
                raise ValueError(f"unknown practice ids: {unknown[:10]}")
            return [self.by_id[i] for i in ids]
        return random.Random(seed).sample(self.practice, min(n, len(self.practice)))


def _structure_summary(s: Structure) -> dict:
    pc = param_count(s)
    return {"layers": list(s.layers), "intermediate_size": s.m, "params": pc["total"],
            "reduction_vs_full": round(pc["reduction_vs_full"], 4)}


def predict_pass(ctx: PrunedContext, req: dict, pass_dir: Path) -> dict:
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
    if which == "both" and not ids:
        n = min(n, 256)
    if "edited" in models:
        structure = ctx.current_structure()  # validate before spending anything
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
    if "edited" in models:
        summary["structure"] = _structure_summary(structure)
    (pass_dir / "predictions.json").write_text(json.dumps(list(per_item.values()), indent=1))
    summary["files"] = {"predictions": str(pass_dir / "predictions.json")}
    summary["note"] = ("predictions.json has per-item gold, predicted label(s) and logits. "
                       "Confusion matrix rows = gold, cols = predicted, in label order.")
    return summary


READ_OUTPUTS = {"cls_hidden", "mean_hidden", "pooled", "logits", "attention", "token_hidden"}


def read_pass(ctx: PrunedContext, req: dict, pass_dir: Path) -> dict:
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
        raise ValueError(f"attention/token_hidden outputs are limited to {MAX_ATTN_TEXTS} texts per pass")
    L = ctx.num_layers(which)  # validates structure.json for the edited model
    structure = ctx.current_structure() if which == "edited" else None
    layers = req.get("layers")
    layers = list(range(L + 1)) if not layers else [int(l) for l in layers]
    if any(l < 0 or l > L for l in layers):
        raise ValueError(f"layers must be in 0..{L} for the {which} model (0 = embedding output, {L} = last encoder layer)")

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

    meta = {"texts": texts, "labels": labels, "ids": [it["id"] for it in items], "model": which, "layers": layers,
            "num_layers": L}
    if structure is not None:
        meta["layer_map"] = {str(k): v for k, v in structure.layer_map().items()}
        meta["structure"] = _structure_summary(structure)
    (pass_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    files["meta"] = str(pass_dir / "meta.json")

    def write_tokens():
        if "tokens" not in files:
            tok = tokenizer()
            toks = [tok.convert_ids_to_tokens(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts]
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

    note = (f"Arrays are float32 .npy in the pass folder; row i corresponds to texts[i] / ids[i] in meta.json. "
            f"Hidden layer index 0 is the embedding output, {L} is the last encoder layer of the {which} model. "
            f"Attention arrays are [n, 12 heads, T, T] for encoder layers 1..{L}.")
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

"""Conventional compression baselines (no LLM) scored by the same evaluator.

- finetune_L{k}[_F{m}]: truncate bert-base to its first k layers (optionally keep m FFN units per
  layer by magnitude) and fine-tune with gradient descent on the practice set.
- naive_<layers>: the start model with layers dropped and nothing refit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wsp.labels import LABEL2ID
from wsp.model import weights as W
from wsp.model.forward import MAX_LEN, pick_device, tokenizer
from wsp.eval.evaluate import DATA_DIR, read_jsonl
from wspc.bundle import edit_manifest, full_change_manifest
from wspc.eval.evaluate import evaluate_checkpoint
from wspc.export import export_checkpoint
from wspc.forward import build_model
from wspc.start import load_start
from wspc.structure import FULL_FFN, FULL_LAYERS, Structure

BASELINES_DIR = W.ROOT / "runs" / "_baselines" / "compress"


def ffn_keep_by_magnitude(weights: dict[str, np.ndarray], layer: int, m: int) -> tuple[int, ...]:
    """Standard magnitude heuristic: score unit i by ||W_in[i]|| * ||W_out[:, i]||, keep the top m."""
    w_in = weights[f"bert.encoder.layer.{layer}.intermediate.dense.weight"]
    w_out = weights[f"bert.encoder.layer.{layer}.output.dense.weight"]
    score = np.linalg.norm(w_in, axis=1) * np.linalg.norm(w_out, axis=0)
    return tuple(sorted(int(i) for i in np.argsort(-score)[:m]))


def truncation_structure(k: int, m: int = FULL_FFN, weights: dict | None = None) -> Structure:
    layers = tuple(range(k))
    if m >= FULL_FFN:
        keep = {l: tuple(range(FULL_FFN)) for l in layers}
    else:
        weights = weights if weights is not None else W.load_base()
        keep = {l: ffn_keep_by_magnitude(weights, l, m) for l in layers}
    return Structure(layers, keep)


def _train(model, texts, labels, *, epochs, lr, batch_size, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    with torch.no_grad():
        model.classifier.weight.normal_(0.0, 0.02)
        model.classifier.bias.zero_()
    model.train()
    tok = tokenizer()
    y_all = torch.tensor(labels)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(texts)
    steps_total = epochs * ((n + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps_total))
    log = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            y = y_all[idx].to(device)
            out = model(**enc, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            loss_sum += out.loss.item() * len(idx); tot += len(idx)
            correct += int((out.logits.argmax(-1) == y).sum())
        log.append({"epoch": ep + 1, "loss": loss_sum / tot, "train_acc": correct / tot})
        print(log[-1], flush=True)
    model.eval()
    return log


def baseline_finetune_trunc(k: int, m: int = FULL_FFN, *, epochs: int = 3, lr: float = 2e-5, batch_size: int = 16,
                            seed: int = 0, name: str | None = None, accuracy_target: float | None = None) -> dict:
    name = name or (f"finetune_L{k}" if m >= FULL_FFN else f"finetune_L{k}_F{m}")
    base = W.load_base()
    structure = truncation_structure(k, m, base)
    device = pick_device()
    model = build_model(base, device, structure=structure)
    practice = read_jsonl(DATA_DIR / "practice.jsonl")
    log = _train(model, [r["text"] for r in practice], [LABEL2ID[r["label"]] for r in practice],
                 epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device)
    weights = {kk: v.detach().cpu().numpy().astype(np.float32) for kk, v in model.state_dict().items() if "position_ids" not in kk}
    prov = {"base_model": "bert-base-uncased", "init_from": None, "method": "truncate + fine-tune",
            "epochs": epochs, "lr": lr, "seed": seed}
    out = BASELINES_DIR / name
    manifest = full_change_manifest(weights, prov, structure)
    export_checkpoint(weights, out / "model", manifest, structure=structure, prune=False)
    return evaluate_checkpoint(out / "model", out / "eval", accuracy_target=accuracy_target,
                               extra={"train_log": log, "epochs": epochs, "lr": lr, "baseline": name})


def baseline_naive_drop(keep_layers: list[int], *, init_from: str = "runs/v1/p50", m: int = FULL_FFN,
                        name: str | None = None, accuracy_target: float | None = None) -> dict:
    name = name or ("naive_L" + "-".join(str(l) for l in keep_layers) + (f"_F{m}" if m < FULL_FFN else ""))
    start, prov = load_start(init_from)
    layers = tuple(keep_layers)
    keep = {l: (tuple(range(FULL_FFN)) if m >= FULL_FFN else ffn_keep_by_magnitude(start, l, m)) for l in layers}
    structure = Structure(layers, keep)
    prov = {**prov, "method": "drop layers, no refit"}
    out = BASELINES_DIR / name
    manifest = edit_manifest(start, {}, prov, structure)
    export_checkpoint(start, out / "model", manifest, structure=structure)
    return evaluate_checkpoint(out / "model", out / "eval", accuracy_target=accuracy_target, extra={"baseline": name})


def baseline_start(init_from: str = "runs/v1/p50", accuracy_target: float | None = None) -> dict:
    """The start model itself, scored through the wspc evaluator (so it carries size and latency)."""
    start, prov = load_start(init_from)
    s = Structure(tuple(range(FULL_LAYERS)), {l: tuple(range(FULL_FFN)) for l in range(FULL_LAYERS)})
    out = BASELINES_DIR / "start"
    export_checkpoint(start, out / "model", edit_manifest(start, {}, {**prov, "method": "start model, unchanged"}, s), structure=s)
    return evaluate_checkpoint(out / "model", out / "eval", accuracy_target=accuracy_target, extra={"baseline": "start"})


def run_suite(accuracy_target: float = 0.978) -> dict[str, dict]:
    results = {}
    jobs = [
        ("start", lambda: baseline_start(accuracy_target=accuracy_target)),
        ("naive_L0-1-2-6-10-11", lambda: baseline_naive_drop([0, 1, 2, 6, 10, 11], accuracy_target=accuracy_target)),
        ("naive_L0-1-2-10-11", lambda: baseline_naive_drop([0, 1, 2, 10, 11], accuracy_target=accuracy_target)),
        ("finetune_L2", lambda: baseline_finetune_trunc(2, accuracy_target=accuracy_target)),
        ("finetune_L4", lambda: baseline_finetune_trunc(4, accuracy_target=accuracy_target)),
        ("finetune_L6", lambda: baseline_finetune_trunc(6, accuracy_target=accuracy_target)),
    ]
    for name, fn in jobs:
        print(f"== baseline: {name}", flush=True)
        s = fn()
        results[name] = {"accuracy": s["test"]["accuracy"], "macro_f1": s["test"]["macro_f1"],
                         "params": s["model"]["total_params"], "meets_target": s.get("meets_target")}
        print(json.dumps(results[name]), flush=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    (BASELINES_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    return results

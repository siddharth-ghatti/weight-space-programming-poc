"""Conventional compression baselines (no LLM) for the reinsurance experiment, scored by the same evaluator.

- ft_base_L{k}[_F{m}]: truncate the MLM base (climatebert-base-f) to its first k layers (optionally keep m
  FFN units per layer by magnitude), new head, fine-tune on the practice paragraphs.
- ft_task_L{k}[_F{m}]: the same starting from the fine-tuned task model itself (keeps its head).
- naive_L...: the start model with layers dropped and nothing refit.
- start: the unmodified start model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wspr.arch import MODELS_DIR, Arch, load_arch
from wspr.bundle import edit_manifest, full_change_manifest
from wspr.eval.evaluate import DATA_DIR, evaluate_checkpoint, read_jsonl
from wspr.export import export_checkpoint
from wspr.forward import MAX_LEN, build_model, pick_device, tokenizer
from wspr.start import BASELINES_DIR, load_checkpoint, load_start
from wspr.structure import Structure

BASE_MLM_DIR = MODELS_DIR / "climatebert-base-f"


def ffn_keep_by_magnitude(weights: dict[str, np.ndarray], arch: Arch, layer: int, m: int) -> tuple[int, ...]:
    w_in = weights[f"{arch.layer_prefix}{layer}.intermediate.dense.weight"]
    w_out = weights[f"{arch.layer_prefix}{layer}.output.dense.weight"]
    score = np.linalg.norm(w_in, axis=1) * np.linalg.norm(w_out, axis=0)
    return tuple(sorted(int(i) for i in np.argsort(-score)[:m]))


def truncation_structure(k: int, m: int, weights: dict[str, np.ndarray], arch: Arch) -> Structure:
    layers = tuple(range(k))
    if m >= arch.ffn:
        keep = {l: tuple(range(arch.ffn)) for l in layers}
    else:
        keep = {l: ffn_keep_by_magnitude(weights, arch, l, m) for l in layers}
    return Structure(layers, keep)


def _train(model, tok, texts, labels, *, epochs, lr, batch_size, seed, device, reinit_head):
    torch.manual_seed(seed); np.random.seed(seed)
    if reinit_head:
        with torch.no_grad():
            model.classifier.dense.weight.normal_(0.0, 0.02); model.classifier.dense.bias.zero_()
            model.classifier.out_proj.weight.normal_(0.0, 0.02); model.classifier.out_proj.bias.zero_()
    model.train()
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


def _mlm_base_as_classifier(arch: Arch) -> dict[str, np.ndarray]:
    """The MLM base's encoder + embeddings with a zero head in the task model's layout."""
    raw = load_checkpoint(BASE_MLM_DIR)
    w = {k: v for k, v in raw.items() if k.startswith(f"{arch.model_prefix}.")}
    w["classifier.dense.weight"] = np.zeros((arch.hidden, arch.hidden), np.float32)
    w["classifier.dense.bias"] = np.zeros((arch.hidden,), np.float32)
    w["classifier.out_proj.weight"] = np.zeros((arch.num_labels, arch.hidden), np.float32)
    w["classifier.out_proj.bias"] = np.zeros((arch.num_labels,), np.float32)
    return w


def baseline_finetune_trunc(k: int, m: int | None = None, *, source: str = "base", epochs: int = 3, lr: float = 2e-5,
                            batch_size: int = 16, seed: int = 0, name: str | None = None, targets: tuple | None = None) -> dict:
    arch, start, prov = load_start("models/climatebert-tcfd")
    m = arch.ffn if m is None else m
    name = name or (f"ft_{source}_L{k}" + (f"_F{m}" if m < arch.ffn else ""))
    if source == "base":
        weights = _mlm_base_as_classifier(arch)
        prov = {"base_model": "climatebert/distilroberta-base-climate-f", "method": "truncate the MLM base + fine-tune"}
    elif source == "task":
        weights = start
        prov = {**prov, "method": "truncate the fine-tuned task model + fine-tune"}
    else:
        raise ValueError("source must be 'base' or 'task'")
    structure = truncation_structure(k, m, weights, arch)
    device = pick_device()
    model = build_model(weights, arch, device, structure=structure)
    tok = tokenizer(arch)
    practice = read_jsonl(DATA_DIR / "practice.jsonl")
    log = _train(model, tok, [r["text"] for r in practice], [arch.label2id[r["label"]] for r in practice],
                 epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device, reinit_head=(source == "base"))
    trained = {kk: v.detach().cpu().numpy().astype(np.float32) for kk, v in model.state_dict().items() if "position_ids" not in kk}
    prov = {**prov, "epochs": epochs, "lr": lr, "seed": seed}
    out = BASELINES_DIR / name
    manifest = full_change_manifest(trained, prov, structure, arch)
    export_checkpoint(trained, out / "model", manifest, arch, structure=structure, prune=False)
    acc_t, f1_t = targets or (None, None)
    return evaluate_checkpoint(out / "model", out / "eval", accuracy_target=acc_t, macro_f1_target=f1_t,
                               extra={"train_log": log, "epochs": epochs, "lr": lr, "baseline": name})


def baseline_naive_drop(keep_layers: list[int], *, m: int | None = None, name: str | None = None, targets: tuple | None = None) -> dict:
    arch, start, prov = load_start("models/climatebert-tcfd")
    m = arch.ffn if m is None else m
    name = name or ("naive_L" + "-".join(str(l) for l in keep_layers) + (f"_F{m}" if m < arch.ffn else ""))
    layers = tuple(keep_layers)
    keep = {l: (tuple(range(arch.ffn)) if m >= arch.ffn else ffn_keep_by_magnitude(start, arch, l, m)) for l in layers}
    structure = Structure(layers, keep)
    out = BASELINES_DIR / name
    manifest = edit_manifest(start, {}, {**prov, "method": "drop layers, no refit"}, structure, arch)
    export_checkpoint(start, out / "model", manifest, arch, structure=structure)
    acc_t, f1_t = targets or (None, None)
    return evaluate_checkpoint(out / "model", out / "eval", accuracy_target=acc_t, macro_f1_target=f1_t, extra={"baseline": name})


def baseline_start(targets: tuple | None = None) -> dict:
    """The unmodified start model, exported through the same path (so it carries size and latency)."""
    arch, start, prov = load_start("models/climatebert-tcfd")
    s = Structure(tuple(range(arch.num_layers)), {l: tuple(range(arch.ffn)) for l in range(arch.num_layers)})
    out = BASELINES_DIR / "start"
    export_checkpoint(start, out / "model", edit_manifest(start, {}, {**prov, "method": "start model, unchanged"}, s, arch), arch, structure=s)
    acc_t, f1_t = targets or (None, None)
    return evaluate_checkpoint(out / "model", out / "eval", accuracy_target=acc_t, macro_f1_target=f1_t, extra={"baseline": "start"})


def run_suite(targets: tuple) -> dict[str, dict]:
    results = {}
    jobs = [
        ("naive_L0-1-2-3", lambda: baseline_naive_drop([0, 1, 2, 3], targets=targets)),
        ("naive_L0-1-2", lambda: baseline_naive_drop([0, 1, 2], targets=targets)),
        ("naive_L0-1-2-3-5", lambda: baseline_naive_drop([0, 1, 2, 3, 5], targets=targets)),
    ]
    for k in (2, 3, 4):
        jobs.append((f"ft_base_L{k}", lambda k=k: baseline_finetune_trunc(k, source="base", targets=targets)))
        jobs.append((f"ft_base_L{k}_e10", lambda k=k: baseline_finetune_trunc(k, source="base", epochs=10, lr=5e-5, name=f"ft_base_L{k}_e10", targets=targets)))
    for k in (3, 4):
        jobs.append((f"ft_task_L{k}_e10", lambda k=k: baseline_finetune_trunc(k, source="task", epochs=10, lr=5e-5, name=f"ft_task_L{k}_e10", targets=targets)))
    for name, fn in jobs:
        print(f"== baseline: {name}", flush=True)
        s = fn()
        results[name] = {"accuracy": s["test"]["accuracy"], "macro_f1": s["test"]["macro_f1"],
                         "params": s["model"]["total_params"], "meets_target": s.get("meets_target")}
        print(json.dumps(results[name]), flush=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    (BASELINES_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    return results

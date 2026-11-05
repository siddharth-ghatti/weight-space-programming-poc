"""Score a checkpoint on the held-out TCFD test split: accuracy, macro-F1, per-class, confusion, size,
latency, and the two-condition target (accuracy and macro-F1 within tolerance of the start model)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from wspr.arch import ROOT, load_arch
from wspr.eval.benchmark import benchmark_latency, model_stats
from wspr.forward import BATCH, MAX_LEN, metrics, pick_device

DATA_DIR = ROOT / "data" / "tcfd"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def assert_test_integrity(data_dir: Path = DATA_DIR) -> None:
    stats = json.loads((data_dir / "stats.json").read_text())
    actual = hashlib.sha256((data_dir / "test.jsonl").read_bytes()).hexdigest()
    if actual != stats["test_sha256"]:
        raise RuntimeError(f"test set hash mismatch: {actual} != {stats['test_sha256']}")


@torch.no_grad()
def predict_dir(model_dir: Path, texts: list[str]) -> np.ndarray:
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().to(device)
    logits = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        logits.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(logits)


def full_metrics(gold: list[int], pred: list[int], labels: tuple[str, ...]) -> dict:
    m = metrics(gold, pred, labels)
    cm = np.array(m["confusion_matrix"])
    support = cm.sum(1)
    per_class = {}
    for c, l in enumerate(labels):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[l] = {"precision": float(p), "recall": float(r), "f1": m["per_class_f1"][l], "support": int(support[c])}
    f1s = np.array([per_class[l]["f1"] for l in labels])
    m["weighted_f1"] = float((f1s * support).sum() / max(1, support.sum()))
    m["per_class"] = per_class
    del m["per_class_f1"]; del m["per_class_recall"]
    return m


def evaluate_checkpoint(model_dir: Path, out_dir: Path, *, data_dir: Path = DATA_DIR, accuracy_target: float | None = None,
                        macro_f1_target: float | None = None, benchmark: bool = True, extra: dict | None = None) -> dict:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    assert_test_integrity(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((model_dir / "config.json").read_text())
    labels = tuple(cfg["id2label"][str(i)] for i in range(len(cfg["id2label"])))
    label2id = {l: i for i, l in enumerate(labels)}
    test = read_jsonl(data_dir / "test.jsonl")
    gold = [label2id[r["label"]] for r in test]
    logits = predict_dir(model_dir, [r["text"] for r in test])
    pred = logits.argmax(-1).tolist()
    try:
        shown = model_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        shown = model_dir.name
    scores: dict = {"model_dir": shown, "test": full_metrics(gold, pred, labels)}
    practice_p = data_dir / "practice.jsonl"
    if practice_p.exists():
        practice = read_jsonl(practice_p)
        pl = predict_dir(model_dir, [r["text"] for r in practice])
        scores["practice"] = full_metrics([label2id[r["label"]] for r in practice], pl.argmax(-1).tolist(), labels)
        scores["generalization_gap"] = scores["practice"]["accuracy"] - scores["test"]["accuracy"]
    man_p = model_dir / "edit_manifest.json"
    if man_p.exists():
        scores["edit_manifest"] = json.loads(man_p.read_text())
    start_arch = load_arch()
    scores["model"] = model_stats(model_dir, start_arch.full_params, start_arch.embedding_params)
    if benchmark:
        texts = [r["text"] for r in test[:64]]
        scores["latency"] = benchmark_latency(model_dir, texts)
    if accuracy_target is not None:
        scores["accuracy_target"] = float(accuracy_target)
        scores["meets_accuracy_target"] = bool(scores["test"]["accuracy"] >= float(accuracy_target))
    if macro_f1_target is not None:
        scores["macro_f1_target"] = float(macro_f1_target)
        scores["meets_macro_f1_target"] = bool(scores["test"]["macro_f1"] >= float(macro_f1_target))
    if accuracy_target is not None or macro_f1_target is not None:
        scores["meets_target"] = bool(scores.get("meets_accuracy_target", True) and scores.get("meets_macro_f1_target", True))
    if extra:
        scores.update(extra)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    with (out_dir / "per_item.jsonl").open("w") as f:
        for r, p, lg in zip(test, pred, logits):
            f.write(json.dumps({"id": r["id"], "gold": r["label"], "pred": labels[p], "correct": label2id[r["label"]] == p,
                                "logits": [round(float(x), 4) for x in lg]}) + "\n")
    (out_dir / "report.md").write_text(render_report(scores, labels))
    return scores


def render_report(s: dict, labels: tuple[str, ...]) -> str:
    t = s["test"]
    lines = [f"# Evaluation: `{s['model_dir']}`", "",
             f"**Test accuracy {t['accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f} | weighted-F1 {t['weighted_f1']:.4f} | n={t['n']}**"]
    if "practice" in s:
        p = s["practice"]
        lines.append(f"Practice accuracy {p['accuracy']:.4f} | macro-F1 {p['macro_f1']:.4f} | gap {s['generalization_gap']:+.4f} "
                     "(practice = the start model's own training data)")
    lines += ["", "## Per-class (test)", "", "| label | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for l in labels:
        c = t["per_class"][l]
        lines.append(f"| {l} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |")
    lines += ["", "## Confusion matrix (rows = gold, cols = pred)", "", "| | " + " | ".join(str(i) for i in range(len(labels))) + " |", "|---|" + "---|" * len(labels)]
    for i, row in enumerate(t["confusion_matrix"]):
        lines.append(f"| {i} {labels[i]} | " + " | ".join(str(x) for x in row) + " |")
    if "edit_manifest" in s and s["edit_manifest"].get("tensors"):
        m = s["edit_manifest"]
        lines += ["", "## Edited tensors", "",
                  f"{m['num_edited_tensors']} tensors, {m['total_changed_entries']:,} entries changed "
                  f"({100 * m['fraction_of_model_changed']:.4f}% of the start model)", "",
                  "| tensor | changed | fraction | ||delta||_F | ||base||_F |", "|---|---|---|---|---|"]
        for tt in m["tensors"]:
            lines.append(f"| {tt['name']} | {tt['changed_entries']:,} | {tt['fraction_changed']:.4f} | {tt['frobenius_delta']:.3f} | {tt['frobenius_base']:.3f} |")
    lines.append(render_size_section(s))
    if s.get("final_answer"):
        lines += ["", "## Editor's final answer", "", s["final_answer"]]
    return "\n".join(lines) + "\n"


def render_size_section(s: dict) -> str:
    m = s["model"]
    full = load_arch().full_params
    lines = ["", "## Size and latency", "",
             f"| total params | non-embedding | layers | FFN width | checkpoint | reduction vs {full:,} | non-embedding reduction |",
             "|---|---|---|---|---|---|---|",
             f"| {m['total_params']:,} | {m['non_embedding_params']:,} | {m['num_hidden_layers']} | {m['intermediate_size']} | {m['checkpoint_mb']} MB | "
             f"{100 * m['reduction_vs_full']:.1f}% | {100 * m.get('non_embedding_reduction_vs_full', 0):.1f}% |"]
    if "latency" in s:
        lat = s["latency"]
        ms = ", ".join(f"batch {b}: {v} ms/paragraph" for b, v in lat["ms_per_paragraph"].items())
        lines.append(f"\nCPU latency ({lat['threads']} threads, {lat['processor']}, max_len {lat['max_len']}, median of {lat['repeats']}): {ms}")
    if "meets_target" in s:
        parts = []
        if "accuracy_target" in s:
            parts.append(f"accuracy >= {100 * s['accuracy_target']:.2f}% ({100 * s['test']['accuracy']:.2f}%: {'ok' if s['meets_accuracy_target'] else 'MISS'})")
        if "macro_f1_target" in s:
            parts.append(f"macro-F1 >= {s['macro_f1_target']:.4f} ({s['test']['macro_f1']:.4f}: {'ok' if s['meets_macro_f1_target'] else 'MISS'})")
        lines.append(f"\nTarget: **{'PASS' if s['meets_target'] else 'FAIL'}** — " + "; ".join(parts))
    return "\n".join(lines) + "\n"


def targets_from_config(cfg: dict) -> tuple[float | None, float | None]:
    return cfg.get("accuracy_target"), cfg.get("macro_f1_target")


def evaluate_run(run_dir: Path) -> dict:
    from wspr.export import export_from_edit_dir, write_model_card

    run_dir = Path(run_dir)
    model_dir = run_dir / "model"
    if not (model_dir / "model.safetensors").exists():
        export_from_edit_dir(run_dir / "edit", model_dir)
    extra = {}
    fa = run_dir / "final_answer.md"
    if fa.exists():
        extra["final_answer"] = fa.read_text()
    st = run_dir / "state.json"
    if st.exists():
        extra["run_state"] = json.loads(st.read_text())
    cfg_p = run_dir / "config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    acc_t, f1_t = targets_from_config(cfg)
    scores = evaluate_checkpoint(model_dir, run_dir / "eval", accuracy_target=acc_t, macro_f1_target=f1_t, extra=extra)
    man_p = model_dir / "edit_manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else None
    line = (f"\n**Held-out test accuracy:** {100 * scores['test']['accuracy']:.2f}% · **macro-F1:** {scores['test']['macro_f1']:.4f}"
            + (f" · target (accuracy >= {100 * acc_t:.2f}%, macro-F1 >= {f1_t:.4f}): {'pass' if scores['meets_target'] else 'fail'}" if acc_t is not None else "") + "\n")
    arch = load_arch(ROOT / manifest["start_model_dir"]) if manifest and manifest.get("start_model_dir") else load_arch()
    write_model_card(model_dir, manifest, arch, extra=line)
    return scores

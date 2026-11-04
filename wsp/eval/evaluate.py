"""Score a checkpoint directory (or a run) on the held-out test set.

Every model in this project (editor runs, baselines) is scored by this same code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from wsp.labels import ID2LABEL, LABEL2ID, LABELS, NUM_LABELS
from wsp.model import weights as W
from wsp.model.forward import MAX_LEN, BATCH, metrics, pick_device

ROOT = W.ROOT
DATA_DIR = ROOT / "data"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_test_integrity(test_path: Path = DATA_DIR / "test.jsonl") -> None:
    stats = json.loads((DATA_DIR / "stats.json").read_text())
    actual = file_sha256(test_path)
    if actual != stats["test_sha256"]:
        raise RuntimeError(f"test set hash mismatch: {actual} != {stats['test_sha256']}")


@torch.no_grad()
def predict_dir(model_dir: Path, texts: list[str]) -> np.ndarray:
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().to(device)
    logits = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i : i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        logits.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(logits)


def full_metrics(gold: list[int], pred: list[int]) -> dict:
    m = metrics(gold, pred)
    cm = np.array(m["confusion_matrix"])
    per_class = {}
    support = cm.sum(1)
    for c in range(NUM_LABELS):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[LABELS[c]] = {"precision": float(p), "recall": float(r), "f1": m["per_class_f1"][LABELS[c]], "support": int(support[c])}
    f1s = np.array([per_class[l]["f1"] for l in LABELS])
    m["weighted_f1"] = float((f1s * support).sum() / max(1, support.sum()))
    m["per_class"] = per_class
    del m["per_class_f1"]
    return m


def evaluate_checkpoint(model_dir: Path, out_dir: Path, *, test_path: Path | None = None,
                        practice_path: Path | None = None, extra: dict | None = None) -> dict:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    test_path = test_path or DATA_DIR / "test.jsonl"
    practice_path = practice_path or DATA_DIR / "practice.jsonl"
    assert_test_integrity(test_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = read_jsonl(test_path)
    gold = [LABEL2ID[r["label"]] for r in test]
    logits = predict_dir(model_dir, [r["text"] for r in test])
    pred = logits.argmax(-1).tolist()
    try:  # record a repo-relative path so reports carry no machine-specific prefix
        shown = model_dir.resolve().relative_to(W.ROOT.resolve()).as_posix()
    except ValueError:
        shown = model_dir.name
    scores = {"model_dir": shown, "test": full_metrics(gold, pred)}

    if practice_path.exists():
        practice = read_jsonl(practice_path)
        pg = [LABEL2ID[r["label"]] for r in practice]
        pl = predict_dir(model_dir, [r["text"] for r in practice])
        scores["practice"] = full_metrics(pg, pl.argmax(-1).tolist())
        scores["generalization_gap"] = scores["practice"]["accuracy"] - scores["test"]["accuracy"]

    manifest_path = model_dir / "edit_manifest.json"
    if manifest_path.exists():
        scores["edit_manifest"] = json.loads(manifest_path.read_text())
    if extra:
        scores.update(extra)

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    with (out_dir / "per_item.jsonl").open("w") as f:
        for r, p, lg in zip(test, pred, logits):
            f.write(json.dumps({"id": r["id"], "text": r["text"], "gold": r["label"], "pred": ID2LABEL[p],
                                "correct": LABEL2ID[r["label"]] == p, "logits": [round(float(x), 4) for x in lg]}) + "\n")
    (out_dir / "report.md").write_text(render_report(scores))
    return scores


def render_report(s: dict) -> str:
    t = s["test"]
    lines = [f"# Evaluation: `{s['model_dir']}`", ""]
    lines.append(f"**Test accuracy {t['accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f} | weighted-F1 {t['weighted_f1']:.4f} | n={t['n']}**")
    if "practice" in s:
        p = s["practice"]
        lines.append(f"Practice accuracy {p['accuracy']:.4f} | macro-F1 {p['macro_f1']:.4f} | gap {s['generalization_gap']:+.4f}")
    lines += ["", "## Per-class (test)", "", "| label | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for l in LABELS:
        c = t["per_class"][l]
        lines.append(f"| {l} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |")
    lines += ["", "## Confusion matrix (rows = gold, cols = pred)", "", "| | " + " | ".join(f"{i}" for i in range(NUM_LABELS)) + " |", "|---|" + "---|" * NUM_LABELS]
    for i, row in enumerate(t["confusion_matrix"]):
        lines.append(f"| {i} {LABELS[i]} | " + " | ".join(str(x) for x in row) + " |")
    if "edit_manifest" in s:
        m = s["edit_manifest"]
        lines += ["", "## Edited tensors", "",
                  f"{m['num_edited_tensors']} tensors, {m['total_changed_entries']:,} entries changed "
                  f"({100 * m['fraction_of_model_changed']:.4f}% of the model)", "",
                  "| tensor | changed | fraction | ||delta||_F | ||base||_F |", "|---|---|---|---|---|"]
        for tt in m["tensors"]:
            lines.append(f"| {tt['name']} | {tt['changed_entries']:,} | {tt['fraction_changed']:.4f} | {tt['frobenius_delta']:.3f} | {tt['frobenius_base']:.3f} |")
    if s.get("final_answer"):
        lines += ["", "## Editor's final answer", "", s["final_answer"]]
    return "\n".join(lines) + "\n"


def evaluate_run(run_dir: Path) -> dict:
    """Evaluate a run directory: export model if needed, then score."""
    from wsp.model.export import export_from_edit_dir

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
    return evaluate_checkpoint(model_dir, run_dir / "eval", extra=extra)

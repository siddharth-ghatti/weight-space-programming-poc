"""runs/d1/compare.md: every arm of the distillation experiment next to the existing rows of runs/compare.md."""
from __future__ import annotations

import json
import re
from pathlib import Path

from wspd.common import ROOT, RUNS


def existing_rows() -> list[dict]:
    """Opus v1/p50, BERT fine-tune, probe and unedited BERT from runs/compare.md (accuracy, macro-F1)."""
    out = []
    want = {"baseline:finetune": "BERT fine-tune (3 ep, local GPU)", "baseline:probe": "BERT linear probe",
            "baseline:unedited": "BERT unedited", "v1/p50": "Opus 5.5 edits BERT weights (v1/p50, 50 passes)"}
    for line in (ROOT / "runs" / "compare.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in want:
            out.append({"name": want[cells[0]], "base": "bert-base-uncased", "acc": float(cells[4]), "f1": float(cells[5]),
                        "cost": cells[3] if cells[3] != "-" else "$0"})
    return out


def collect(runs: Path = RUNS) -> list[dict]:
    rows = []
    dirs = [d for d in sorted(runs.glob("*")) if not d.name.startswith("smoke_")] + sorted((runs / "ablation").glob("*"))
    for d in dirs:
        f = d / "eval" / "scores.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        g, sc, ret, ret0 = s.get("generation", {}), s.get("scored", {}), s.get("retention", {}), s.get("retention_before", {})
        e = s.get("edit", {}) or {}
        rows.append({"name": ("ablation: " if d.parent.name == "ablation" else "") + d.name, "kind": s.get("kind", "-"), "acc_gen": g.get("accuracy"), "f1_gen": g.get("macro_f1"),
                     "acc_scored": sc.get("accuracy"), "f1_scored": sc.get("macro_f1"), "ece": sc.get("ece"),
                     "sel80": (sc.get("selective") or {}).get("accuracy"), "agree": sc.get("agree_teacher"),
                     "teacher_acc": sc.get("teacher_accuracy"),
                     "wiki_ppl": ret.get("wiki_ppl"), "wiki_ppl0": ret0.get("wiki_ppl"), "arc": ret.get("arc_acc"), "arc0": ret0.get("arc_acc"),
                     "caps": ret.get("capitals_acc"), "contam": ret.get("contamination"),
                     "changed": e.get("weights_changed", s.get("weights_changed")), "wall": s.get("wall_seconds"), "seed": s.get("seed")})
    return rows


def _f(x, pct=False, nd=4):
    if x is None:
        return "-"
    return f"{100 * x:.2f}%" if pct else f"{x:.{nd}f}"


def render(rows: list[dict], ext: list[dict]) -> str:
    lines = ["# Distillation results (d1)", "",
             "Insurance benchmark: 1,496 practice sentences (1,456 train the teacher and the edit, 40 kept back for doses and "
             "model selection), 1,779 held-out test sentences never used for anything. Reader: Qwen3.5-0.8B, frozen. "
             "\"generated\" reads the first word the detached model writes; \"scored\" takes the argmax over the eight answer words "
             "at the answer position.", "",
             "| arm | test acc (generated) | macro-F1 (generated) | test acc (scored) | ECE | selective acc @80% | agrees with teacher | "
             "WikiText-2 ppl (before -> after) | ARC-Easy | capitals | contaminated | weights changed | wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ppl = f"{_f(r['wiki_ppl0'], nd=2)} -> {_f(r['wiki_ppl'], nd=2)}" if r.get("wiki_ppl") else "-"
        ch = f"{int(r['changed']):,}" if r.get("changed") else "-"
        wall = f"{r['wall']:.0f}" if r.get("wall") else "-"
        lines.append(f"| {r['name']} | {_f(r['acc_gen'], True)} | {_f(r['f1_gen'])} | {_f(r['acc_scored'], True)} | {_f(r['ece'], nd=3)} | "
                     f"{_f(r['sel80'], True)} | {_f(r['agree'], True)} | {ppl} | {_f(r['arc'], True)} | {_f(r['caps'], True)} | "
                     f"{_f(r['contam'], True)} | {ch} | {wall} |")
    lines += ["", "Existing rows on the same test set (different base model: BERT-base, 110M):", "",
              "| model | test acc | macro-F1 | API cost |", "|---|---|---|---|"]
    for r in ext:
        lines.append(f"| {r['name']} | {100 * r['acc']:.2f}% | {r['f1']:.4f} | {r['cost']} |")
    return "\n".join(lines) + "\n"


def write_compare(runs: Path = RUNS) -> Path:
    out = runs / "compare.md"
    out.write_text(render(collect(runs), existing_rows()))
    return out

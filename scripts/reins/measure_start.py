#!/usr/bin/env python
"""Score the unmodified start model on the held-out test split and write the derived targets into
configs/reins.yaml (accuracy - 0.01, macro-F1 - 0.02). Also produces runs/_baselines/reins/start/.

    uv run python scripts/reins/measure_start.py [--acc-margin 0.01] [--f1-margin 0.02]
"""
import argparse, re
from pathlib import Path
from wspr.arch import ROOT
from wspr.eval.baselines import baseline_start

p = argparse.ArgumentParser()
p.add_argument("--acc-margin", type=float, default=0.01)
p.add_argument("--f1-margin", type=float, default=0.02)
p.add_argument("--no-write", action="store_true")
a = p.parse_args()
s = baseline_start()
acc, f1 = s["test"]["accuracy"], s["test"]["macro_f1"]
acc_t, f1_t = round(acc - a.acc_margin, 4), round(f1 - a.f1_margin, 4)
print(f"start model: test accuracy {acc:.4f}  macro-F1 {f1:.4f}  per-class recall "
      f"{ {k: round(v['recall'], 3) for k, v in s['test']['per_class'].items()} }")
print(f"targets: accuracy >= {acc_t:.4f}  macro-F1 >= {f1_t:.4f}")
if not a.no_write:
    cfg = ROOT / "configs" / "reins.yaml"
    txt = cfg.read_text()
    txt = re.sub(r"^accuracy_target:.*$", f"accuracy_target: {acc_t}             # start accuracy {acc:.4f} - {a.acc_margin}", txt, flags=re.M)
    txt = re.sub(r"^macro_f1_target:.*$", f"macro_f1_target: {f1_t}             # start macro-F1 {f1:.4f} - {a.f1_margin}", txt, flags=re.M)
    cfg.write_text(txt)
    # re-score the start row with the targets so its scores.json carries pass/fail like every other row
    baseline_start(targets=(acc_t, f1_t))
    print(f"wrote targets to {cfg}")

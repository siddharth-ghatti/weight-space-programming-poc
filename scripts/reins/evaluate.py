#!/usr/bin/env python
"""Evaluate a run dir (runs/r1/p25) or a checkpoint dir (.../model) with the wspr evaluator."""
import argparse
from pathlib import Path
from wspr.eval.evaluate import evaluate_checkpoint, evaluate_run

p = argparse.ArgumentParser()
p.add_argument("path", type=Path)
p.add_argument("--out", type=Path)
p.add_argument("--acc-target", type=float)
p.add_argument("--f1-target", type=float)
a = p.parse_args()
if (a.path / "model.safetensors").exists():
    s = evaluate_checkpoint(a.path, a.out or a.path.parent / "eval", accuracy_target=a.acc_target, macro_f1_target=a.f1_target)
else:
    s = evaluate_run(a.path)
m = s["model"]
print(f"TEST accuracy {s['test']['accuracy']:.4f}  macro-F1 {s['test']['macro_f1']:.4f}  params {m['total_params']:,}  "
      f"latency {s.get('latency', {}).get('ms_per_paragraph')}  target {s.get('meets_target')}")

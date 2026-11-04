#!/usr/bin/env python
"""Evaluate a run dir (runs/p10) or a checkpoint dir (runs/_baselines/probe/model) on the held-out test set."""
import argparse, json
from pathlib import Path
from wsp.eval.evaluate import evaluate_run, evaluate_checkpoint

p = argparse.ArgumentParser()
p.add_argument("path", type=Path)
p.add_argument("--out", type=Path, help="output dir (default: <path>/eval or <path>/../eval for a model dir)")
a = p.parse_args()
if (a.path / "model.safetensors").exists():
    s = evaluate_checkpoint(a.path, a.out or a.path.parent / "eval")
else:
    s = evaluate_run(a.path)
print(json.dumps({k: s["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "n")}, indent=2))

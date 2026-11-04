#!/usr/bin/env python
"""Compute reference models: uv run python scripts/run_baselines.py [--only probe]"""
import argparse, json
from wsp.eval import baselines as B

p = argparse.ArgumentParser()
p.add_argument("--only", choices=["unedited", "probe", "finetune"])
a = p.parse_args()
if a.only:
    s = {"unedited": B.baseline_unedited, "probe": B.baseline_probe, "finetune": B.baseline_finetune}[a.only]()
    print(json.dumps({"accuracy": s["test"]["accuracy"], "macro_f1": s["test"]["macro_f1"]}, indent=2))
else:
    B.run_all()

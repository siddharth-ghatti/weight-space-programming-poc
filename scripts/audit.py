#!/usr/bin/env python
"""Post-hoc audit of a run: uv run python scripts/audit.py runs/p10"""
import argparse, json
from pathlib import Path
from wsp.harness.audit import audit_run

p = argparse.ArgumentParser()
p.add_argument("run_dir", type=Path)
a = p.parse_args()
print(json.dumps(audit_run(a.run_dir), indent=2))

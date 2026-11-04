#!/usr/bin/env python
"""Export a run's edit bundle as a runnable HF checkpoint: uv run python scripts/export.py runs/p10 [--out DIR]"""
import argparse
from pathlib import Path
from wsp.model.export import export_from_edit_dir

p = argparse.ArgumentParser()
p.add_argument("run_dir", type=Path)
p.add_argument("--out", type=Path)
a = p.parse_args()
out = export_from_edit_dir(a.run_dir / "edit", a.out or a.run_dir / "model")
print(f"exported to {out}")

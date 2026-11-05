#!/usr/bin/env python
"""Rebuild a pruned checkpoint from the public start checkpoint + this run's edit + structure.

  uv run python scripts/reins/apply_edit.py runs/r1/p25/edit --out /tmp/rebuilt [--check runs/r1/p25/model]
"""
import argparse, json
from pathlib import Path
from safetensors import safe_open
from wspr.start import tensor_sha256
from wspr.export import export_from_edit_dir

p = argparse.ArgumentParser()
p.add_argument("edit_dir", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--check", type=Path)
a = p.parse_args()
man = json.loads((a.edit_dir / "manifest.json").read_text())
st = man.get("structure", {})
print(f"chain: {man.get('diff_relative_to')} + {a.edit_dir} + structure(k={len(st.get('layers', []))}, m={st.get('m')}) -> {man.get('exported_params'):,} params")
export_from_edit_dir(a.edit_dir, a.out)
print(f"rebuilt checkpoint at {a.out}")
if a.check:
    def hashes(d):
        with safe_open(str(d / "model.safetensors"), "np") as f:
            return {k: tensor_sha256(f.get_tensor(k)) for k in f.keys()}
    print("identical" if hashes(a.out) == hashes(a.check) else "DIFFERENT")

#!/usr/bin/env python
"""Rebuild an edited model bit-for-bit from the public bert-base-uncased weights + an edit bundle.

uv run python scripts/apply_edit.py runs/p10/edit --out /tmp/rebuilt [--check runs/p10/model]
Verifies every sha256 in manifest.json; with --check, also compares against an existing export.
"""
import argparse, hashlib
from pathlib import Path
import numpy as np
from safetensors import safe_open
from wsp.model import weights as W
from wsp.model.export import export_checkpoint

p = argparse.ArgumentParser()
p.add_argument("edit_dir", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--check", type=Path, help="existing checkpoint dir to compare tensor hashes against")
a = p.parse_args()
base = W.load_base()
edits = W.load_edit_bundle(a.edit_dir, base, verify=True)
print(f"verified {len(edits)} edited tensors against manifest hashes")
import json
manifest = json.loads((a.edit_dir / "manifest.json").read_text())
export_checkpoint(W.apply_edits(base, edits), a.out, manifest)
print(f"rebuilt checkpoint at {a.out}")
if a.check:
    mism = 0
    with safe_open(str(a.out / "model.safetensors"), "np") as f1, safe_open(str(a.check / "model.safetensors"), "np") as f2:
        for k in f1.keys():
            if W.sha256(f1.get_tensor(k)) != W.sha256(f2.get_tensor(k)):
                mism += 1; print("MISMATCH", k)
    print("identical" if mism == 0 else f"{mism} tensors differ")

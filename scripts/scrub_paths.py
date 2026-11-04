#!/usr/bin/env python
"""Replace machine-specific absolute paths in run logs and reports with placeholders.

Run before publishing: `uv run python scripts/scrub_paths.py` (add --check to only report).
The editor sometimes echoes its absolute working-folder path (tracebacks, print(WORK)); this rewrites
the repo root to `<repo>` and the home directory to `<home>` in every text file under runs/, data/ and docs/.
"""
import argparse
import os
from pathlib import Path

from wsp.model.weights import ROOT

p = argparse.ArgumentParser()
p.add_argument("--check", action="store_true", help="report files that would change, do not write")
a = p.parse_args()

root = str(ROOT.resolve())
home = str(Path.home())
subs = [(root, "<repo>"), (root.replace("/", "\\/"), "<repo>"), (home, "<home>")]
skip_suffix = {".npy", ".npz", ".safetensors", ".pyc"}
changed = 0
for base in ("runs", "data", "docs"):
    for f in (ROOT / base).rglob("*"):
        if not f.is_file() or f.suffix in skip_suffix or "/work/" in f.as_posix() or "/model/" in f.as_posix():
            continue
        try:
            t = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        new = t
        for old, rep in subs:
            new = new.replace(old, rep)
        if new != t:
            changed += 1
            print(("would change " if a.check else "scrubbed ") + f.relative_to(ROOT).as_posix())
            if not a.check:
                f.write_text(new)
print(f"{changed} files" + (" would change" if a.check else " changed"))

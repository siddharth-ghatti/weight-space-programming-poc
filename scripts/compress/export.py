#!/usr/bin/env python
"""Export a pruning run's edit bundle to a checkpoint dir."""
import argparse
from pathlib import Path
from wspc.export import export_from_edit_dir

p = argparse.ArgumentParser()
p.add_argument("edit_dir", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
print(export_from_edit_dir(a.edit_dir, a.out))

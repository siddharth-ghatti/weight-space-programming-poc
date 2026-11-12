"""Rebuild the baked checkpoint from the base model + edit bundle and compare tensor by tensor.

    uv run python scripts/distill/check_rebuild.py runs/d1/direct_s0
"""
import json
import sys
from pathlib import Path

import wspd  # noqa: F401
from wspd.export import rebuild_check

d = Path(sys.argv[1])
r = rebuild_check(d / "model", d / "edit")
print(json.dumps(r, indent=1))
sys.exit(0 if r["identical"] else 1)

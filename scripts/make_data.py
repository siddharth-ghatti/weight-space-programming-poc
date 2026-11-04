#!/usr/bin/env python
"""Synthesize the dataset: uv run python scripts/make_data.py [--per-label 5] [--workers 4]"""
import argparse
from wsp.data.synth import build_dataset

p = argparse.ArgumentParser()
p.add_argument("--model", default="claude-opus-5-5")
p.add_argument("--per-label", type=int, default=5, help="sentences per label per batch (84 batches)")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--n-practice", type=int, default=1500)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
build_dataset(model=a.model, per_label=a.per_label, workers=a.workers, n_practice=a.n_practice, seed=a.seed)

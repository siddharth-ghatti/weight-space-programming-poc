#!/usr/bin/env python
"""Classify sentences with an exported checkpoint: uv run python scripts/predict.py --model runs/p10/model "The policy excludes flood damage."""
import argparse
from transformers import pipeline

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("texts", nargs="+")
a = p.parse_args()
clf = pipeline("text-classification", model=a.model, top_k=3)
for t, preds in zip(a.texts, clf(a.texts)):
    print(f"{t}\n  -> " + ", ".join(f"{x['label']} ({x['score']:.3f})" for x in preds))

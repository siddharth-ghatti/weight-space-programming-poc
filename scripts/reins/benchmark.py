#!/usr/bin/env python
"""Size and CPU latency of a checkpoint dir: uv run python scripts/reins/benchmark.py runs/r1/p25/model"""
import argparse, json
from pathlib import Path
from wspr.arch import load_arch
from wspr.eval.evaluate import DATA_DIR, read_jsonl
from wspr.eval.benchmark import benchmark_latency, model_stats

p = argparse.ArgumentParser()
p.add_argument("model_dir", type=Path)
p.add_argument("--batch-sizes", default="1,16")
p.add_argument("--repeats", type=int, default=7)
p.add_argument("--n", type=int, default=64)
p.add_argument("--threads", type=int, default=4)
a = p.parse_args()
arch = load_arch()
texts = [r["text"] for r in read_jsonl(DATA_DIR / "test.jsonl")[:a.n]]
out = {"model": model_stats(a.model_dir, arch.full_params, arch.embedding_params),
       "latency": benchmark_latency(a.model_dir, texts, tuple(int(b) for b in a.batch_sizes.split(",")), repeats=a.repeats, threads=a.threads)}
print(json.dumps(out, indent=2))

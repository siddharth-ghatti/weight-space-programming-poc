"""Size and CPU latency of a checkpoint directory (paragraph inputs)."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 512


def model_stats(model_dir: Path, full_params: int, full_embedding_params: int | None = None) -> dict:
    model_dir = Path(model_dir)
    total = emb = 0
    with safe_open(str(model_dir / "model.safetensors"), "np") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            if sl.get_dtype() not in ("F32", "F16", "BF16", "F64"):
                continue
            n = 1
            for d in sl.get_shape():
                n *= d
            total += n
            if ".embeddings." in k:
                emb += n
    cfg = json.loads((model_dir / "config.json").read_text())
    out = {"total_params": total, "non_embedding_params": total - emb, "num_hidden_layers": cfg.get("num_hidden_layers"),
           "intermediate_size": cfg.get("intermediate_size"),
           "checkpoint_mb": round((model_dir / "model.safetensors").stat().st_size / 1e6, 1),
           "reduction_vs_full": round(1 - total / full_params, 4)}
    if full_embedding_params is not None:
        out["non_embedding_reduction_vs_full"] = round(1 - (total - emb) / (full_params - full_embedding_params), 4)
    return out


@torch.no_grad()
def benchmark_latency(model_dir: Path, texts: list[str], batch_sizes=(1, 16), repeats: int = 7,
                      warmup: int = 2, threads: int = 4, max_len: int = MAX_LEN) -> dict:
    fast = os.environ.get("WSP_FAST_EVAL") == "1"
    if fast:
        texts, repeats, warmup = texts[:4], 1, 0
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().to("cpu")
    out = {"threads": threads, "n_paragraphs": len(texts), "repeats": repeats, "max_len": max_len, "torch": torch.__version__,
           "processor": platform.processor() or platform.machine(), "ms_per_paragraph": {}}
    for b in batch_sizes:
        batches = [texts[i:i + b] for i in range(0, len(texts), b)]
        encs = [tok(ch, padding=True, truncation=True, max_length=max_len, return_tensors="pt") for ch in batches]
        for _ in range(warmup):
            for enc in encs:
                model(**enc)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for enc in encs:
                model(**enc)
            times.append((time.perf_counter() - t0) / len(texts) * 1000)
        out["ms_per_paragraph"][str(b)] = round(statistics.median(times), 3)
    return out

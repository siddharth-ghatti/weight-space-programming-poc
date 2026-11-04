"""Size and CPU latency of a checkpoint directory."""

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

from wsp.model.forward import MAX_LEN
from wspc.structure import FULL_PARAMS


def model_stats(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    total = emb = 0
    with safe_open(str(model_dir / "model.safetensors"), "np") as f:
        for k in f.keys():
            n = 1
            for d in f.get_slice(k).get_shape():
                n *= d
            total += n
            if k.startswith("bert.embeddings."):
                emb += n
    cfg = json.loads((model_dir / "config.json").read_text())
    return {
        "total_params": total,
        "non_embedding_params": total - emb,
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "intermediate_size": cfg.get("intermediate_size"),
        "checkpoint_mb": round((model_dir / "model.safetensors").stat().st_size / 1e6, 1),
        "reduction_vs_full": round(1 - total / FULL_PARAMS, 4),
    }


@torch.no_grad()
def benchmark_latency(model_dir: Path, texts: list[str], batch_sizes=(1, 32), repeats: int = 7,
                      warmup: int = 2, threads: int = 4) -> dict:
    fast = os.environ.get("WSP_FAST_EVAL") == "1"
    if fast:
        texts, repeats, warmup = texts[:8], 1, 0
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().to("cpu")
    out = {"threads": threads, "n_sentences": len(texts), "repeats": repeats, "torch": torch.__version__,
           "processor": platform.processor() or platform.machine(), "ms_per_sentence": {}}
    for b in batch_sizes:
        batches = [texts[i:i + b] for i in range(0, len(texts), b)]
        encs = [tok(ch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt") for ch in batches]
        for _ in range(warmup):
            for enc in encs:
                model(**enc)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for enc in encs:
                model(**enc)
            times.append((time.perf_counter() - t0) / len(texts) * 1000)
        out["ms_per_sentence"][str(b)] = round(statistics.median(times), 3)
    return out

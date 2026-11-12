"""Splits of the insurance benchmark and the forgetting-suite data (WikiText-2, ARC-Easy) as parquet."""
from __future__ import annotations

import json
import random
from pathlib import Path

from wsp.eval.evaluate import assert_test_integrity, read_jsonl
from wsp.labels import LABEL2ID
from wspd.common import CACHE_DIR, DATA_DIR


def load_rows() -> tuple[list[dict], list[dict]]:
    """Practice (the editor's and teacher's data) and the held-out test set, with integrity check."""
    assert_test_integrity(DATA_DIR / "test.jsonl")
    practice = read_jsonl(DATA_DIR / "practice.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")
    for r in practice + test:
        r["gold"] = LABEL2ID[r["label"]]
    return practice, test


def split_practice(practice: list[dict], n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified by label: n_val practice rows kept back from the edit (doses, layer and width are chosen on
    them); the rest train the teacher and the edit. The test set is never part of this."""
    rng = random.Random(seed)
    by: dict[int, list[dict]] = {}
    for r in practice:
        by.setdefault(r["gold"], []).append(r)
    val, train = [], []
    for k in sorted(by):
        rs = list(by[k]); rng.shuffle(rs)
        n = round(len(rs) * n_val / len(practice))
        val += rs[:n]; train += rs[n:]
    rng.shuffle(val); rng.shuffle(train)
    return train, val


def _parquet(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(CACHE_DIR)))


def load_wikitext(n: int, min_chars: int = 400) -> list[str]:
    """WikiText-2 (raw, test split) paragraphs of at least `min_chars`, section headers skipped."""
    import pyarrow.parquet as pq
    t = pq.read_table(_parquet("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"))
    out = []
    for x in t.column("text").to_pylist():
        s = x.strip()
        if len(x) > min_chars and not s.startswith("="):
            out.append(s)
        if len(out) >= n:
            break
    return out


def load_arc(n: int, seed: int = 0) -> list[dict]:
    """ARC-Easy test questions: {question, labels, texts, answer}."""
    import pyarrow.parquet as pq
    t = pq.read_table(_parquet("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet")).to_pylist()
    rng = random.Random(seed); rng.shuffle(t)
    return [{"question": x["question"], "labels": list(x["choices"]["label"]), "texts": list(x["choices"]["text"]),
             "answer": x["answerKey"]} for x in t[:n]]


def save_split(run_dir: Path, train: list[dict], val: list[dict], seed: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps({"seed": seed, "train": [r["id"] for r in train], "val": [r["id"] for r in val]}))

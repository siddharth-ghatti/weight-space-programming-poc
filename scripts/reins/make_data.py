"""Build data/tcfd/{practice.jsonl,test.jsonl,stats.json} from climatebert/tcfd_recommendations.

Practice = the dataset's train split (the start model's own training data); test = its test split.
Rows labeled "none" (not climate-related) are dropped because the start model has no such class.
Labels are written with the start model's names (config id2label order).

    uv run python scripts/reins/make_data.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "tcfd"
DATASET = "climatebert/tcfd_recommendations"
LICENSE = "CC BY-NC-SA 4.0 (non-commercial; research use)"
MODEL_DIR = ROOT / "models" / "climatebert-tcfd"


def main() -> None:
    api = HfApi()
    info = api.dataset_info(DATASET)
    files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    model_labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"dataset": DATASET, "dataset_commit": info.sha, "license": LICENSE, "model_labels": model_labels,
             "note": "practice = train split of the dataset = the start model's own fine-tuning data; test split never shown to the editor",
             "dropped_label": "none"}
    for split, prefix in (("train", "p"), ("test", "t")):
        fn = [f for f in files if split in Path(f).name]
        if len(fn) != 1:
            raise SystemExit(f"expected one {split} parquet, found {fn}")
        path = hf_hub_download(DATASET, fn[0], repo_type="dataset", revision=info.sha)
        table = pq.read_table(path)
        rows = table.to_pylist()
        # ClassLabel names are stored in the parquet schema metadata written by `datasets`
        meta = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}"))
        dataset_names = meta["info"]["features"]["label"]["names"]
        kept, dropped, counts = [], 0, {}
        for r in rows:
            lab = r["label"]
            name = dataset_names[int(lab)] if isinstance(lab, int) else str(lab)
            if name == "none":
                dropped += 1
                continue
            if name not in model_labels:
                raise SystemExit(f"label {name!r} not in model labels {model_labels}")
            kept.append({"id": f"{prefix}{len(kept):04d}", "text": r["text"].strip(), "label": name})
            counts[name] = counts.get(name, 0) + 1
        dest = OUT / ("practice.jsonl" if split == "train" else "test.jsonl")
        dest.write_text("".join(json.dumps(k) + "\n" for k in kept))
        stats[f"{split}_source_file"] = fn[0]
        stats["practice" if split == "train" else "test"] = len(kept)
        stats[f"{'practice' if split == 'train' else 'test'}_labels"] = {l: counts.get(l, 0) for l in model_labels}
        stats[f"{split}_dropped_none"] = dropped
        print(split, "->", dest.name, len(kept), "rows;", dropped, "'none' dropped;", counts)
    stats["test_sha256"] = hashlib.sha256((OUT / "test.jsonl").read_bytes()).hexdigest()
    stats["practice_sha256"] = hashlib.sha256((OUT / "practice.jsonl").read_bytes()).hexdigest()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k.endswith("labels")}, indent=1))


if __name__ == "__main__":
    main()

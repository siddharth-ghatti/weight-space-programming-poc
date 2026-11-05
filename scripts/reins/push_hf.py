#!/usr/bin/env python
"""Push an exported r1 checkpoint dir (RoBERTa tokenizer files) to the Hugging Face Hub.

    uv run python scripts/reins/push_hf.py runs/r1/p50/model --repo <user>/<name> [--private]
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
            "special_tokens_map.json", "README.md", "edit_manifest.json", "structure.json", "param_count.json"]

p = argparse.ArgumentParser()
p.add_argument("model_dir", type=Path)
p.add_argument("--repo", required=True)
p.add_argument("--private", action="store_true")
p.add_argument("--message", default="weight-space-programming pruned checkpoint")
a = p.parse_args()
missing = [f for f in REQUIRED if not (a.model_dir / f).exists()]
if missing:
    raise SystemExit(f"{a.model_dir} is missing {missing}")
api = HfApi()
api.create_repo(a.repo, private=a.private, exist_ok=True)
url = api.upload_folder(folder_path=str(a.model_dir), repo_id=a.repo, commit_message=a.message)
print(f"pushed {a.model_dir} -> https://huggingface.co/{a.repo}")

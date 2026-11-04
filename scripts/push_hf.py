#!/usr/bin/env python
"""Push an exported checkpoint folder (runs/<run>/model) to the Hugging Face Hub.

    uv run python scripts/push_hf.py runs/v1/p25/model --repo <user>/<name> [--private]

Needs a write token: `huggingface-cli login` or HF_TOKEN in the environment. Nothing is pushed
unless you run this script.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("model_dir", type=Path)
p.add_argument("--repo", required=True, help="e.g. username/bert-insurance-wsp-p25")
p.add_argument("--private", action="store_true")
p.add_argument("--message", default="weight-space-programming checkpoint")
a = p.parse_args()

required = ["model.safetensors", "config.json", "tokenizer.json", "vocab.txt"]
missing = [f for f in required if not (a.model_dir / f).exists()]
if missing:
    raise SystemExit(f"{a.model_dir} is missing {missing}; export it first (scripts/export.py)")

api = HfApi()
api.create_repo(a.repo, private=a.private, exist_ok=True)
url = api.upload_folder(folder_path=str(a.model_dir), repo_id=a.repo, commit_message=a.message)
print(f"pushed {a.model_dir} -> https://huggingface.co/{a.repo}")

#!/usr/bin/env python
"""Push every exported checkpoint (runs/<version>/<run>/model and runs/_baselines/<name>/model) to
the Hugging Face Hub, one repo per checkpoint, filling in the model card with the repo id, the test
score and the source-code link.

    uv run python scripts/push_all_hf.py --user <hf-user> [--private] [--prefix wsp-bert-insurance] [--only v1/p50 ...]
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from wsp.model.weights import ROOT

p = argparse.ArgumentParser()
p.add_argument("--user", required=True)
p.add_argument("--prefix", default="wsp-bert-insurance")
p.add_argument("--private", action="store_true")
p.add_argument("--source", default="https://github.com/siddharth-ghatti/weight-space-programming-poc")
p.add_argument("--only", nargs="*", help="run keys like v1/p50 or _baselines/finetune")
a = p.parse_args()

runs = sorted(ROOT.glob("runs/*/*/model"))
api = HfApi()
for model_dir in runs:
    key = model_dir.parent.relative_to(ROOT / "runs").as_posix()  # v1/p50, _baselines/finetune
    if a.only and key not in a.only:
        continue
    slug = key.replace("_baselines/", "baseline-").replace("/", "-")
    repo = f"{a.user}/{a.prefix}-{slug}"
    scores_p = model_dir.parent / "eval" / "scores.json"
    s = json.loads(scores_p.read_text()) if scores_p.exists() else {}
    acc = s.get("test", {}).get("accuracy")
    f1 = s.get("test", {}).get("macro_f1")
    card_p = model_dir / "README.md"
    card = card_p.read_text().replace("<this repo>", repo)
    extra = f"\n**Run:** `{key}`"
    if acc is not None:
        extra += f" · **held-out test accuracy:** {100 * acc:.2f}% · **macro-F1:** {f1:.4f}"
    extra += f"\n\n**Source, logs and protocol:** {a.source}\n"
    if "**Run:**" not in card:
        card = card.rstrip("\n") + "\n" + extra
    card_p.write_text(card)
    print(f"-> {repo} ({key}, acc={acc})", flush=True)
    api.create_repo(repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo, commit_message=f"{key}: weight-space-programming checkpoint")
    print(f"   done https://huggingface.co/{repo}", flush=True)
print("all done")

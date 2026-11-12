#!/usr/bin/env python
"""Push the d1 artifacts to one Hugging Face Hub repo: the baked checkpoint at the root (stock
`AutoModelForCausalLM`), the three edit bundles under edit/s{0,1,2}/ and the three Jev-like teacher
heads under teacher/s{0,1,2}/. The model card gains a section describing those folders and a footer
with the run, the score and the source link.

    uv run python scripts/distill/push_hf.py --repo <user>/<name> [--private] [--seeds 0 1 2]

Needs a write token: `huggingface-cli login` or HF_TOKEN in the environment. Nothing is pushed
unless you run this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

import wspd  # noqa: F401
from wspd.common import ROOT

MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "README.md", "edit_manifest.json", "LICENSE"]
EDIT_FILES = ["edit.npz", "manifest.json", "rebuild_check.json"]
TEACHER_FILES = ["head.npz", "teacher.json"]


def card_extra(repo: str, run: str, scores: dict, seeds: list[int], source: str) -> str:
    gen = scores.get("generation", {})
    return f"""
## What else is in this repo

- `edit/s<k>/edit.npz` + `manifest.json`: the exact diff for seed k (the 1024 column indices and the `[1024, 1024]` fp32
  update `D` of `layers.<layer>.mlp.down_proj`), with the base commit and the hash of `D`. `rebuild_check.json` records that
  base checkpoint + bundle rebuilt this folder tensor for tensor. Apply it with `wspd.edit.ColumnEdit` from the source repo
  (`scripts/distill/check_rebuild.py` is the reference).
- `teacher/s<k>/head.npz` + `teacher.json`: the Jev-like teacher the edit was distilled from, a standardized multinomial
  logistic head (`mean`, `scale`, `coef [8, 2048]`, `intercept`) over two features read from the frozen base model at the
  answer position (`final_last`: input of the final norm; `L16_mean`: mean over sentence tokens of the layer-16 residual),
  with its calibration temperature and abstain threshold. Load with `wspd.teacher.Teacher.load`. It scores 98.8% on the same
  test set and is the ceiling the edit aims at.

Seeds published: {", ".join(str(s) for s in seeds)}. Only seed 0 is baked into the weights at the root.

**Run:** `d1/{run}` · **held-out test accuracy:** {100 * gen.get("accuracy", 0):.2f}% · **macro-F1:** {gen.get("macro_f1", 0):.4f}

**Source, logs and protocol:** {source}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. <user>/wsp-qwen3.5-0.8b-insurance-d1-direct")
    p.add_argument("--run", default="direct_s0", help="arm whose baked model goes at the repo root")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--private", action="store_true")
    p.add_argument("--source", default="https://github.com/siddharth-ghatti/weight-space-programming-poc")
    a = p.parse_args()

    d1 = ROOT / "runs" / "d1"
    model_dir = d1 / a.run / "model"
    missing = [f for f in MODEL_FILES if not (model_dir / f).exists()]
    if missing:
        raise SystemExit(f"{model_dir} is missing {missing}; run scripts/distill/run.py first")
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(model_dir / f)) for f in MODEL_FILES if f != "README.md"]
    for s in a.seeds:
        for f in EDIT_FILES:
            src = d1 / f"direct_s{s}" / "edit" / f
            if not src.exists():
                raise SystemExit(f"missing {src}")
            ops.append(CommitOperationAdd(path_in_repo=f"edit/s{s}/{f}", path_or_fileobj=str(src)))
        for f in TEACHER_FILES:
            src = d1 / f"teacher_s{s}" / f
            if not src.exists():
                raise SystemExit(f"missing {src}")
            ops.append(CommitOperationAdd(path_in_repo=f"teacher/s{s}/{f}", path_or_fileobj=str(src)))

    scores = json.loads((d1 / a.run / "eval" / "scores.json").read_text())
    card = (model_dir / "README.md").read_text()
    if "## What else is in this repo" not in card:
        card = card.rstrip("\n") + "\n" + card_extra(a.repo, a.run, scores, a.seeds, a.source)
        (model_dir / "README.md").write_text(card)
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))

    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    print(f"uploading {len(ops)} files to {a.repo} ...", flush=True)
    api.create_commit(repo_id=a.repo, operations=ops, commit_message=f"d1/{a.run}: closed-form weight edit of Qwen3.5-0.8B, edit bundles and teacher heads")
    print(f"pushed -> https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()

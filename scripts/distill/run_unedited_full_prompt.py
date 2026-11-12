"""Reference row: the unedited reader prompted with the long option descriptions (2.4x the tokens of the prompt the
edit uses), scored and generated on the test set. Shows what prompting alone buys before any weight is changed.

    uv run python scripts/distill/run_unedited_full_prompt.py [--smoke]
"""
from __future__ import annotations

import argparse
import time

import wspd  # noqa: F401
from wspd.common import ROOT
from wspd.edit import Bench
from wspd.pipeline import Log, attach_targets, evaluate_arm, load_config, run_dir, splits_for, teacher_dir
from wspd.reader import Student


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    cfg = load_config()
    splits = splits_for(0, cfg, a.smoke); attach_targets(splits, teacher_dir(0, a.smoke))
    out = run_dir("unedited_full_prompt", None, a.smoke); out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt"); T0 = time.perf_counter()
    student = Student(ROOT / cfg["model_dir"], variant="full")
    bench = Bench(student, {"train": splits["train"][:64], "val": splits["val"], "test": splits["test"]}, variant="full",
                  chunk=cfg.get("chunk", 32), post_splits=())
    evaluate_arm(bench, splits, None, None, out.name, out, kind="unedited", seed=None, cfg=cfg,
                 extra={"prompt_variant": "full", "weights_changed": 0, "wall_seconds": time.perf_counter() - T0}, log=log)
    log(f"done {out.relative_to(ROOT)} in {(time.perf_counter() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()

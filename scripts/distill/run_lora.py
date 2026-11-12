"""The gradient-descent arm for one split seed: LoRA on every MLP matrix, same loss, rows and selection rule as the
direct edit. Trains in fp32 (backward through the gated-DeltaNet reference path is NaN in fp16 on Metal), bakes,
then loads the baked weights into the fp16 reader so the test and forgetting measurements match every other arm.

    uv run python scripts/distill/run_lora.py --seed 0 [--smoke] [--train-device cpu|mps]
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch

import wspd  # noqa: F401
from wspd.common import ROOT
from wspd.lora import LoRA, train_lora
from wspd.pipeline import SMOKE, Log, attach_targets, build, evaluate_arm, load_config, make_bench, run_dir, splits_for, teacher_dir
from wspd.reader import Student, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-device", default=None, help="device for the fp32 training copy (default: the reader's)")
    a = ap.parse_args()
    cfg = load_config()
    lc = dict(cfg["lora"])
    if a.smoke:
        lc["steps"], lc["eval_every"] = SMOKE["lora_steps"], SMOKE["lora_eval_every"]
    out = run_dir("lora", a.seed, a.smoke); out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    T0 = time.perf_counter()
    # 1. the fp32 training copy, alone in memory (a second model alongside it pushes Metal into swapping)
    splits = splits_for(a.seed, cfg, a.smoke); attach_targets(splits, teacher_dir(a.seed, a.smoke))
    tdev = a.train_device or pick_device()
    s32 = Student(ROOT / cfg["model_dir"], device=tdev, dtype=torch.float32, variant=cfg.get("variant", "words"))
    b32 = make_bench(s32, {"train": splits["train"], "val": splits["val"]}, cfg)
    log(f"training copy in fp32 on {tdev}; train {b32.n['train']} val {b32.n['val']} test {len(splits['test'])}")
    lora = LoRA(s32.model, rank=lc["rank"], alpha=2 * lc["rank"], seed=a.seed)
    n_lora = lora.n_params()
    log(f"LoRA rank {lc['rank']} on gate/up/down of every layer: {n_lora:,} trainable parameters")
    R = train_lora(b32, lora, lc["steps"], lc["lr"], lc["batch"], cfg["guard_weight"], lc["eval_every"], seed=a.seed,
                   log=lambda s, h, m: log(f"  step {s:3d} {h['ms']:6.0f} ms/step  train gap {h['gap']:.3f} guard KL {h['guard_kl']:.4f} | "
                                           f"val agree {m['val_agree']:.3f} gap {m['val_gap']:.3f} acc {m['val_acc']:.3f}"))
    ms = sorted(h["ms"] for h in R["hist"])
    log(f"trained in {R['train_seconds']:.0f}s (median {ms[len(ms) // 2] if ms else 0:.0f} ms/step, {R['skipped']} skipped), kept step {R['best_step']}")
    B = lora.bake()
    log(f"merged into {B['weights_in_targets']:,} weights ({B['weights_changed']:,} changed) across {B['matrices']} matrices in {B['bake_ms']:.0f} ms")
    baked = {k: v.detach().to("cpu", torch.float16) for k, v in s32.model.state_dict().items()}
    del b32, lora, s32; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # 2. the fp16 measuring copy: every baseline (guard, post-answer, WikiText top-64) is the unedited model's, then the baked weights go in
    student, bench, splits, ret = build(a.seed, cfg, a.smoke)
    ret_before = ret.measure()
    baked = {k: v.to(student.dtype) for k, v in baked.items()}
    missing, unexpected = student.model.load_state_dict(baked, strict=False)
    assert not unexpected and all("rotary" in k or "inv_freq" in k for k in missing), (missing[:5], unexpected[:5])
    del baked; gc.collect()
    log(f"baked weights loaded into the {student.dtype} reader on {student.device}")
    extra = {"lora": {**lc, "params": n_lora, "best_step": R["best_step"], "train_seconds": R["train_seconds"], "train_device": tdev,
                      "evals": R["evals"], "skipped": R["skipped"], "bake": B},
             "weights_changed": B["weights_changed"], "wall_seconds": time.perf_counter() - T0}
    evaluate_arm(bench, splits, ret, ret_before, out.name, out, kind="lora", seed=a.seed, cfg=cfg, extra=extra, log=log)
    (out / "config.json").write_text(json.dumps({**cfg, "lora": lc}, indent=1))
    log(f"done {out.relative_to(ROOT)} in {(time.perf_counter() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()

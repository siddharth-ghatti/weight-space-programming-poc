"""Ablations of the direct edit at the main run's layer/width/lambda (seed 0), each measured on the test set (scored
at the answer position) and the forgetting suite, then removed: random columns, no protection, hard gold labels,
25% and 50% of the training rows, a random change of the same size.

    uv run python scripts/distill/run_ablations.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

import wspd  # noqa: F401
from wspd.common import ROOT
from wspd.edit import Bench, ColumnEdit, DirectEditor, random_like, smoothed_one_hot
from wspd.pipeline import SMOKE, Log, build, evaluate_arm, load_config, make_bench, run_dir
from wspd.reader import down_proj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="run a single ablation by name (e.g. no_protection) and leave the others as they are")
    a = ap.parse_args()
    cfg = load_config()
    if a.smoke:
        cfg["rounds"] = SMOKE["rounds"]
    main_scores = json.loads((run_dir("direct", a.seed, a.smoke) / "eval" / "scores.json").read_text())
    E = main_scores["edit"]
    LAYER, WIDTH, LAM = E["layer"], E["n_columns"], E["lambda"]
    out_root = run_dir("ablation", None, a.smoke)
    log = Log(out_root / "log.txt")
    T0 = time.perf_counter()
    student, bench, splits, ret = build(a.seed, cfg, a.smoke)
    ret_before = ret.measure()
    log(f"ablations at layer {LAYER}, {WIDTH} columns, lambda {LAM}")

    def ablate(name, b, lam=None, width=None, cols=None, cap=None, guard=None):
        if a.only and name != a.only:
            return
        lam = LAM if lam is None else lam; width = WIDTH if width is None else width
        t = time.perf_counter()
        ed = DirectEditor(b, LAYER, width, lam, cols=cols)
        try:
            r = ed.run(cfg["rounds"], cfg["doses"], cfg["guard_weight"] if guard is None else guard,
                       cfg["guard_cap"] if cap is None else cap, eval_splits=("val",), forgetting=True,
                       log=lambda h: log(f"  [{name}] round {h['round']}: dose {h['dose']}{' refused ' + str(h['refused']) if h['refused'] else ''}, "
                                         f"val gap {h['val_gap']:.3f} agree {h['val_agree']:.3f} acc {h['val_acc']:.3f} | guard KL {h['guard_kl']:.4f} "
                                         f"unseen KL {h['holdout_kl']:.4f} ppl {h.get('holdout_ppl', float('nan')):.1f} capitals {h['capitals_acc']:.2f}"))
            extra = {"ablation": name, "rounds": r["best_round"], "trajectory": r["hist"],
                     "edit": {"layer": LAYER, "n_columns": len(ed.cols), "lambda": lam,
                     "best_round": r["best_round"], "weights_in_edit": int(ed.edit.D.numel()),
                     "weights_changed": int((ed.edit.D != 0).sum().item()), "wall_seconds": time.perf_counter() - t},
                     "wall_seconds": time.perf_counter() - t}
            evaluate_arm(b, splits, ret, ret_before, f"ablation: {name}", out_root / name, kind="ablation", seed=a.seed, cfg=cfg,
                         generation=False, extra=extra, log=log)
        finally:
            ed.edit.remove()

    ablate("full_method", bench)
    n_cols = down_proj(student.model, LAYER).weight.shape[1]
    ablate("random_columns", bench, cols=sorted(random.Random(a.seed + 1).sample(range(n_cols), WIDTH)))
    ablate("no_protection", bench, lam=0.0, cap=float("inf"), guard=0.0)
    hard = {s: [dict(r, teacher=smoothed_one_hot(r["gold"])) for r in rows] for s, rows in splits.items()}
    hard["test"] = splits["test"]
    hb = make_bench(student, hard, cfg)          # same prompts, so the keys and covariance are the main bench's
    hb._kc = bench._kc
    ablate("hard_labels", hb)
    for frac in cfg["ablation_fracs"]:
        n = max(20, int(len(splits["train"]) * frac))
        sub = {"train": splits["train"][:n], "val": splits["val"], "test": splits["test"]}
        sb = make_bench(student, sub, cfg)
        sb.kc_subset(bench, n)
        ablate(f"rows_{int(100 * frac)}pct", sb)
    # a random change of the same size as the main run's, on its columns
    z = __import__("numpy").load(run_dir("direct", a.seed, a.smoke) / "edit" / "edit.npz")
    e = ColumnEdit(student.model, LAYER, z["cols"].tolist())
    e.D.copy_(random_like(torch.as_tensor(z["D"]).to(e.D.device)))
    t = time.perf_counter()
    evaluate_arm(bench, splits, ret, ret_before, "ablation: random_change_same_size", out_root / "random_change_same_size",
                 kind="ablation", seed=a.seed, cfg=cfg, generation=False,
                 extra={"ablation": "random_change_same_size", "edit": {"layer": LAYER, "n_columns": len(z["cols"]), "lambda": None,
                        "weights_changed": int(e.D.numel()), "wall_seconds": time.perf_counter() - t}, "wall_seconds": time.perf_counter() - t}, log=log)
    e.remove()
    log(f"ablations done in {(time.perf_counter() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()

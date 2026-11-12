"""Fit the Jev-like teacher (calibrated heads on the frozen reader) for one split seed, score it on the test set,
and write the target probabilities every arm distils from.

    uv run python scripts/distill/fit_teacher.py --seed 0 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import wspd  # noqa: F401  (MPS fallback switch)
from wsp.labels import LABELS
from wspd.common import ROOT, prompt
from wspd.eval.evaluate import write_scores
from wspd.pipeline import Log, load_config, splits_for, teacher_dir
from wspd.reader import Student
from wspd.teacher import fit_teacher, read_features, score_teacher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    out = teacher_dir(a.seed, a.smoke); out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    splits = splits_for(a.seed, cfg, a.smoke)
    student = Student(ROOT / cfg["model_dir"], variant=cfg.get("variant", "words"))
    t0 = time.perf_counter()
    feats = {}
    for split, rows in splits.items():
        t = time.perf_counter()
        feats[split] = read_features(student, [prompt(r["text"], cfg.get("variant", "words")) for r in rows])
        log(f"features {split}: {len(rows)} rows, {sorted(feats[split])} in {time.perf_counter() - t:.0f}s")
    y = {s: np.array([r["gold"] for r in rows]) for s, rows in splits.items()}
    log("5-fold CV on the training rows:")
    teacher = fit_teacher(feats["train"], y["train"], seed=a.seed, log=log)
    teacher.save(out)
    rep = teacher.report
    log(f"chosen {rep['chosen']}  oof acc {rep['oof_accuracy']:.4f}  T={rep['temperature']:.3f}  "
        f"oof ECE {rep['oof_ece_after_calibration']:.3f}  selective@80 {rep['oof_selective']['accuracy']:.4f}")
    val = score_teacher(teacher, feats["val"], y["val"]); test = score_teacher(teacher, feats["test"], y["test"])
    log(f"teacher val acc {100 * val['accuracy']:.2f}%  test acc {100 * test['accuracy']:.2f}%  macro-F1 {test['macro_f1']:.4f}  "
        f"ECE {test['ece']:.3f}  selective@80 {100 * test['selective']['accuracy']:.2f}%  abstain {100 * test['abstain_rate']:.1f}%")
    targets = {}
    for split, rows in splits.items():
        P = teacher.probs(feats[split])
        for r, p in zip(rows, P):
            targets[r["id"]] = [float(x) for x in p]
    (out / "targets.json").write_text(json.dumps(targets))
    test["teacher_accuracy"] = test["accuracy"]; test["agree_teacher"] = 1.0
    write_scores(out, {"name": out.name, "kind": "teacher", "seed": a.seed, "scored": test, "val": val, "fit_report": rep,
                       "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
                       "wall_seconds": time.perf_counter() - t0, "decide_example": teacher.decide({k: v[:1] for k, v in feats["test"].items()})[0]})
    log(f"saved {out.relative_to(ROOT)} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Shared bootstrap for the scripts: config, splits, teacher targets, student, bench, retention, arm evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from wspd.common import ROOT, RUNS, prompt
from wspd.data import load_arc, load_rows, load_wikitext, split_practice
from wspd.edit import Bench
from wspd.eval.evaluate import score_at_answer_position, score_generation, write_scores
from wspd.reader import Student
from wspd.retention import Retention

CONFIG = ROOT / "configs" / "distill.yaml"
SMOKE = {"n_train": 64, "n_val": 16, "n_test": 32, "sweep_layers": [21, 23], "widths": [256], "lambdas": [1.0],
         "sweep_rounds": 1, "rounds": 2, "n_wiki": 4, "n_arc": 8, "lora_steps": 4, "lora_eval_every": 2}


def load_config(path: Path = CONFIG, **overrides) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def teacher_dir(seed: int, smoke: bool = False) -> Path:
    return RUNS / (f"{'smoke_' if smoke else ''}teacher_s{seed}")


def run_dir(name: str, seed: int | None, smoke: bool = False) -> Path:
    return RUNS / (f"{'smoke_' if smoke else ''}{name}" + (f"_s{seed}" if seed is not None else ""))


def splits_for(seed: int, cfg: dict, smoke: bool = False) -> dict[str, list[dict]]:
    practice, test = load_rows()
    train, val = split_practice(practice, cfg["n_val"], seed)
    if smoke:
        train, val, test = train[:SMOKE["n_train"]], val[:SMOKE["n_val"]], test[:SMOKE["n_test"]]
    return {"train": train, "val": val, "test": test}


def attach_targets(splits: dict[str, list[dict]], tdir: Path) -> None:
    """Teacher probabilities per row id (from fit_teacher.py) -> r["teacher"]."""
    targets = json.loads((tdir / "targets.json").read_text())
    for rows in splits.values():
        for r in rows:
            r["teacher"] = targets[r["id"]]


def build(seed: int, cfg: dict, smoke: bool = False, retention: bool = True):
    splits = splits_for(seed, cfg, smoke)
    attach_targets(splits, teacher_dir(seed, smoke))
    student = Student(ROOT / cfg["model_dir"], variant=cfg.get("variant", "words"))
    bench = make_bench(student, splits, cfg)
    ret = None
    if retention:
        rc = cfg["retention"]
        n_wiki, n_arc = (SMOKE["n_wiki"], SMOKE["n_arc"]) if smoke else (rc["n_wiki"], rc["n_arc"])
        ret = Retention(student, load_wikitext(n_wiki), load_arc(n_arc, seed), gen_tokens=rc["gen_tokens"])
    return student, bench, splits, ret


def make_bench(student: Student, splits, cfg: dict, **kw) -> Bench:
    return Bench(student, splits, variant=cfg.get("variant", "words"), chunk=cfg.get("chunk", 32), **kw)


def evaluate_arm(bench: Bench, splits, ret: Retention | None, ret_before: dict | None, name: str, out: Path, *,
                 kind: str, seed: int | None, cfg: dict, generation: bool = True, extra: dict | None = None, log=print) -> dict:
    """Score one arm on the test set the same way as every other arm and write eval/scores.json + report.md."""
    t = time.perf_counter()
    scores = {"name": name, "kind": kind, "seed": seed, "spelling": bench.spelling, "n_train": bench.n["train"],
              "n_val": bench.n["val"], "n_test": bench.n["test"]}
    scores["scored"] = score_at_answer_position(bench, "test")
    scores["val"] = {k[4:]: v for k, v in bench.evaluate(("val",), forgetting=False).items() if k.startswith("val_")}
    if generation:
        gen = bench.generation_test(splits["test"], cfg.get("gen_tokens", 4))
        scores["generation"] = score_generation([r["gold"] for r in splits["test"]], gen)
        scores["generation_texts_sample"] = gen["texts"][:12]
    if ret is not None:
        after = ret.measure()
        scores["retention"] = after
        if ret_before is not None:
            scores["retention_before"] = ret_before
            scores["forgot"] = Retention.forgot(ret_before, after)
    scores["eval_seconds"] = time.perf_counter() - t
    if extra:
        scores.update(extra)
    write_scores(out, scores)
    g = scores.get("generation", {})
    log(f"[{name}] scored acc {100 * scores['scored']['accuracy']:.2f}%  " + (f"gen acc {100 * g['accuracy']:.2f}%  " if g else "") +
        f"agree {100 * scores['scored']['agree_teacher']:.1f}%  ECE {scores['scored']['ece']:.3f}"
        + (f"  | wiki ppl {scores['retention']['wiki_ppl']:.2f} ARC {100 * scores['retention']['arc_acc']:.1f}% caps "
           f"{100 * scores['retention']['capitals_acc']:.0f}% contam {100 * scores['retention']['contamination']:.0f}%" if ret else "")
        + f"  ({scores['eval_seconds']:.0f}s)")
    return scores


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + "\n"); self.f.flush()

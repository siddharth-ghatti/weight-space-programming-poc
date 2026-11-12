"""The direct edit for one split seed: unedited baseline (seed 0), layer sweep, width/lambda grid, the main run,
bake, test evaluation, forgetting suite, edit bundle and (optionally) the baked checkpoint.

    uv run python scripts/distill/run.py --seed 0 [--smoke] [--export-model] [--layer L --width W --lam X]
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import wspd  # noqa: F401
from wspd.common import ROOT
from wspd.edit import DirectEditor
from wspd.export import export_checkpoint, model_card, rebuild_check, save_edit_bundle
from wspd.pipeline import SMOKE, Log, build, evaluate_arm, load_config, make_bench, run_dir


def sweep(bench, jobs, rounds, cfg, log, cache=None):
    """Run each (layer, width, lambda) job; results already in `cache` (a json file) are reused on a restart."""
    out = []
    done = {(x["layer"], x["width"], x["lam"]): x for x in json.loads(cache.read_text())} if cache and cache.exists() else {}
    for layer, width, lam in jobs:
        if (layer, width, lam) in done:
            out.append(done[(layer, width, lam)]); log(f"  -> layer {layer} width {width} lambda {lam}: reused from {cache.name}"); continue
        t0 = time.perf_counter()
        ed = DirectEditor(bench, layer, width, lam)
        try:
            r = ed.run(rounds, cfg["doses"], cfg["guard_weight"], cfg["guard_cap"], eval_splits=("val",), forgetting=True,
                       log=lambda h: log(f"  [layer {layer}, {width} cols, lambda {lam}] round {h['round']}: dose {h['dose']}"
                                         f"{' refused ' + str(h['refused']) if h['refused'] else ''}, val gap {h['val_gap']:.3f}, "
                                         f"agree {h['val_agree']:.3f}, acc {h['val_acc']:.3f}, guard KL {h['guard_kl']:.4f}, "
                                         f"unseen text KL {h['holdout_kl']:.4f}, capitals {h['capitals_acc']:.2f}"))
        finally:
            ed.edit.remove()
        last = r["hist"][r["best_round"]]
        out.append({"layer": layer, "width": len(ed.cols), "lam": lam, "rounds": r["best_round"], "read_s": ed.read_s,
                    "val_gap": last["val_gap"], "agree": last["val_agree"], "val_acc": last["val_acc"],
                    "holdout_kl": last["holdout_kl"], "capitals": last["capitals_acc"], "seconds": time.perf_counter() - t0})
        log(f"  -> layer {layer} width {width} lambda {lam}: kept round {r['best_round']}, agree {last['val_agree']:.3f}, "
            f"gap {last['val_gap']:.3f} ({time.perf_counter() - t0:.0f}s)")
        if cache is not None:
            done[(layer, width, lam)] = out[-1]; cache.write_text(json.dumps(list(done.values()), indent=1))
    return out


def free_mps() -> None:
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export-model", action="store_true", help="also save the baked 1.5 GB checkpoint")
    ap.add_argument("--layer", type=int); ap.add_argument("--width", type=int); ap.add_argument("--lam", type=float)
    ap.add_argument("--skip-unedited", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.smoke:
        cfg.update({k: SMOKE[k] for k in ("sweep_layers", "widths", "lambdas", "sweep_rounds", "rounds")})
    out = run_dir("direct", a.seed, a.smoke); out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    T0 = time.perf_counter()
    student, bench, splits, ret = build(a.seed, cfg, a.smoke)
    log(f"reader {student.model_dir.name} on {student.device} {student.dtype}, {student.n_params / 1e6:.0f}M params; "
        f"train {bench.n['train']} val {bench.n['val']} test {bench.n['test']}; spelling {bench.spelling}")
    before = bench.evaluate(("val", "test"))
    ret_before = ret.measure()
    log(f"unedited: test scored acc {before['test_acc']:.4f} agree {before['test_agree']:.3f} gap {before['test_gap']:.3f} "
        f"mass {before['test_mass']:.2f} | teacher acc {before['test_teacher_acc']:.4f} | wiki ppl {ret_before['wiki_ppl']:.2f} "
        f"ARC {ret_before['arc_acc']:.3f} capitals {ret_before['capitals_acc']:.2f}")
    un = run_dir("unedited", None, a.smoke)
    if not a.skip_unedited and (a.seed == 0 or a.smoke) and not (un / "eval" / "scores.json").exists():
        evaluate_arm(bench, splits, ret, ret_before, un.name, un, kind="unedited", seed=a.seed, cfg=cfg,
                     extra={"wall_seconds": 0.0, "weights_changed": 0}, log=log)

    if a.layer is None:
        # the layer and the width/lambda are chosen on a subset of the training rows (reads are the whole cost)
        n_sw = min(cfg.get("sweep_rows", 256), bench.n["train"])
        sb = make_bench(student, {"train": splits["train"][:n_sw], "val": splits["val"]}, cfg) if n_sw < bench.n["train"] else bench
        log(f"layer sweep on {n_sw} training rows:")
        SW = sweep(sb, [(L, cfg["sweep_width"], 1.0) for L in cfg["sweep_layers"]], cfg["sweep_rounds"], cfg, log, out / "sweep.json")
        clean = [x for x in SW if x["capitals"] >= before["capitals_acc"] - 0.05] or SW
        best = max(clean, key=lambda x: (x["agree"], -x["val_gap"]))
        LAYER = best["layer"]
        log(f"layer {LAYER} chosen (highest agreement on the kept-back rows, gap as tie-break)")
        jobs = [(LAYER, w, lam) for w in cfg["widths"] for lam in cfg["lambdas"]
                if not (w == cfg["sweep_width"] and lam == 1.0) and (w == cfg["sweep_width"] or lam == 1.0 or w == cfg["widths"][-1])]
        log("width x lambda grid:")
        GR = sweep(sb, jobs, cfg["sweep_rounds"], cfg, log, out / "sweep.json") + [best]
        if sb is not bench:
            del sb
        free_mps()
        ok = [x for x in GR if x["capitals"] >= before["capitals_acc"] - 0.05] or GR
        top = max(x["agree"] for x in ok)
        one_row = 1.0 / max(1, bench.n["val"])
        pick = max([x for x in ok if x["agree"] >= top - one_row - 1e-9], key=lambda x: (x["lam"], x["agree"], -x["val_gap"]))
        WIDTH, LAM = pick["width"], pick["lam"]
        log(f"width {WIDTH}, lambda {LAM} chosen: the most protection within one kept-back row of the best agreement")
    else:
        SW, GR = [], []
        LAYER, WIDTH, LAM = a.layer, a.width or cfg["sweep_width"], a.lam if a.lam is not None else 1.0

    log(f"main run: layer {LAYER}, {WIDTH} columns, lambda {LAM}, up to {cfg['rounds']} rounds")
    t = time.perf_counter()
    ed = DirectEditor(bench, LAYER, WIDTH, LAM)
    RUN = ed.run(cfg["rounds"], cfg["doses"], cfg["guard_weight"], cfg["guard_cap"], eval_splits=("val",),
                 log=lambda h: log(f"  round {h['round']}: solve {h['solve_ms']:.1f} ms, dose {h['dose']}"
                                   f"{' refused ' + str(h['refused']) if h['refused'] else ''}, train gap before {h['train_gap_before']:.3f}, "
                                   f"val gap {h['val_gap']:.3f} agree {h['val_agree']:.3f} acc {h['val_acc']:.3f} | guard KL {h['guard_kl']:.4f} "
                                   f"unseen KL {h['holdout_kl']:.4f} capitals {h['capitals_acc']:.2f}"))
    direct_seconds = time.perf_counter() - t
    TRAJ = RUN["hist"]
    edit_ms = sum(h["solve_ms"] + h["write_ms"] for h in TRAJ[1:])
    log(f"kept round {RUN['best_round']}; {len(TRAJ) - 1} rounds in {direct_seconds:.0f}s; solve+write {edit_ms:.1f} ms in all")

    free_mps()
    hooked = bench.evaluate(("test",), forgetting=False)
    BAKE = ed.edit.bake()
    detached = bench.evaluate(("test",), forgetting=False)
    log(f"baked: {BAKE['weights_changed']} of {BAKE['weights_in_edit']} weights changed in {BAKE['bake_ms']:.2f} ms; "
        f"gap hooked {hooked['test_gap']:.4f} vs baked {detached['test_gap']:.4f} (fp16 rounding)")
    bench._kc.clear()
    man = save_edit_bundle(out / "edit", LAYER, ed.cols, ed.edit.D, student.provenance,
                           {"lambda": LAM, "doses": [h["dose"] for h in TRAJ[1:]], "best_round": RUN["best_round"],
                            "weights_changed": BAKE["weights_changed"], "model_params": student.n_params, "seed": a.seed,
                            "spelling": bench.spelling})
    extra = {"edit": {"layer": LAYER, "n_columns": len(ed.cols), "lambda": LAM, "best_round": RUN["best_round"], "edit_ms": edit_ms,
                      "weights_changed": BAKE["weights_changed"], "weights_in_edit": BAKE["weights_in_edit"],
                      "direct_seconds": direct_seconds, "wall_seconds": time.perf_counter() - T0, "D_sha256": man["D_sha256"]},
             "sweep": SW, "grid": GR, "trajectory": TRAJ, "hooked_vs_baked_gap": [hooked["test_gap"], detached["test_gap"]],
             "unedited": before, "wall_seconds": time.perf_counter() - T0}
    scores = evaluate_arm(bench, splits, ret, ret_before, out.name, out, kind="direct", seed=a.seed, cfg=cfg, extra=extra, log=log)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    if a.export_model:
        card = model_card(man, scores, out.name)
        export_checkpoint(student, out / "model", man, card)
        chk = rebuild_check(out / "model", out / "edit", ROOT / cfg["model_dir"])
        (out / "edit" / "rebuild_check.json").write_text(json.dumps(chk, indent=1))
        log(f"exported {out.name}/model; rebuild from base + edit identical: {chk['identical']}")
    log(f"done {out.relative_to(ROOT)} in {(time.perf_counter() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()

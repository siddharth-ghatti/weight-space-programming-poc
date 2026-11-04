"""Score a pruned checkpoint: accuracy (same code as the training harness) + size + latency + target."""

from __future__ import annotations

import json
from pathlib import Path

from wsp.eval.evaluate import DATA_DIR, evaluate_checkpoint as _wsp_evaluate, read_jsonl
from wspc.eval.benchmark import benchmark_latency, model_stats
from wspc.structure import FULL_PARAMS


def evaluate_checkpoint(model_dir: Path, out_dir: Path, *, accuracy_target: float | None = None,
                        benchmark: bool = True, extra: dict | None = None) -> dict:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    scores = _wsp_evaluate(model_dir, out_dir, extra=extra)
    scores["model"] = model_stats(model_dir)
    if benchmark:
        texts = [r["text"] for r in read_jsonl(DATA_DIR / "test.jsonl")[:64]]
        scores["latency"] = benchmark_latency(model_dir, texts)
    if accuracy_target is not None:
        scores["accuracy_target"] = float(accuracy_target)
        scores["meets_target"] = bool(scores["test"]["accuracy"] >= float(accuracy_target))
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    with (out_dir / "report.md").open("a") as f:
        f.write(render_size_section(scores))
    return scores


def render_size_section(s: dict) -> str:
    m = s["model"]
    lines = ["", "## Size and latency", "",
             f"| total params | non-embedding | layers | FFN width | checkpoint | reduction vs {FULL_PARAMS:,} |",
             "|---|---|---|---|---|---|",
             f"| {m['total_params']:,} | {m['non_embedding_params']:,} | {m['num_hidden_layers']} | {m['intermediate_size']} | {m['checkpoint_mb']} MB | {100 * m['reduction_vs_full']:.1f}% |"]
    if "latency" in s:
        lat = s["latency"]
        ms = ", ".join(f"batch {b}: {v} ms/sentence" for b, v in lat["ms_per_sentence"].items())
        lines.append(f"\nCPU latency ({lat['threads']} threads, {lat['processor']}, median of {lat['repeats']}): {ms}")
    if "meets_target" in s:
        lines.append(f"\nAccuracy target {100 * s['accuracy_target']:.1f}%: **{'PASS' if s['meets_target'] else 'FAIL'}** "
                     f"(test accuracy {100 * s['test']['accuracy']:.2f}%)")
    return "\n".join(lines) + "\n"


def evaluate_run(run_dir: Path) -> dict:
    from wspc.export import export_from_edit_dir, write_model_card

    run_dir = Path(run_dir)
    model_dir = run_dir / "model"
    if not (model_dir / "model.safetensors").exists():
        export_from_edit_dir(run_dir / "edit", model_dir)
    extra = {}
    fa = run_dir / "final_answer.md"
    if fa.exists():
        extra["final_answer"] = fa.read_text()
    st = run_dir / "state.json"
    if st.exists():
        extra["run_state"] = json.loads(st.read_text())
    cfg_p = run_dir / "config.json"
    target = json.loads(cfg_p.read_text()).get("accuracy_target") if cfg_p.exists() else None
    scores = evaluate_checkpoint(model_dir, run_dir / "eval", accuracy_target=target, extra=extra)
    man_p = model_dir / "edit_manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else None
    acc_line = (f"\n**Held-out test accuracy:** {100 * scores['test']['accuracy']:.2f}% · **macro-F1:** "
                f"{scores['test']['macro_f1']:.4f}" + (f" · target {100 * target:.1f}%: {'pass' if scores['meets_target'] else 'fail'}" if target else "") + "\n")
    write_model_card(model_dir, manifest, extra=acc_line)
    return scores

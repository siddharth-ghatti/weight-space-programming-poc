#!/usr/bin/env python
"""Render runs/r1/compare.md: reinsurance pruning runs + compression baselines."""
import json
import yaml
from wspr.arch import ROOT, load_arch

RUNS = ROOT / "runs"
arch = load_arch()
cfg = yaml.safe_load((ROOT / "configs" / "reins.yaml").read_text())
rows = []
paths = sorted(RUNS.glob("r1/*/eval/scores.json")) + sorted(RUNS.glob("_baselines/reins/*/eval/scores.json"))
for scores in paths:
    s = json.loads(scores.read_text())
    run = scores.parent.parent
    rel = run.relative_to(RUNS).as_posix()
    st = s.get("run_state", {})
    m = s.get("model", {})
    lat = s.get("latency", {}).get("ms_per_paragraph", {})
    rows.append({
        "name": rel.replace("_baselines/reins/", "baseline:"),
        "passes": f"{st.get('passes_used', '-')}/{st.get('passes', '-')}" if st else "-",
        "code_runs": st.get("code_runs", "-") if st else "-",
        "cost": f"${st['cost_usd']:.2f}" if st else "-",
        "acc": s["test"]["accuracy"], "f1": s["test"]["macro_f1"],
        "gov": s["test"]["per_class"].get("governance", {}).get("recall", float("nan")),
        "params": m.get("total_params"), "nonemb": m.get("non_embedding_params"),
        "shape": f"{m.get('num_hidden_layers', '-')}L/{m.get('intermediate_size', '-')}" if m else "-",
        "mb": m.get("checkpoint_mb", "-"),
        "b1": lat.get("1", "-"), "b16": lat.get("16", "-"),
        "target": "-" if "meets_target" not in s else ("pass" if s["meets_target"] else "fail"),
    })
lines = ["# Reinsurance pruning results (r1)", "",
         f"Start model: {arch.model_dir.name} ({arch.full_params:,} params, {arch.embedding_params:,} in embeddings = the floor). "
         f"Target: test accuracy >= {100 * (cfg.get('accuracy_target') or 0):.2f}% and macro-F1 >= {cfg.get('macro_f1_target')} "
         "(start model minus 1.0 point / 0.02). Latency is CPU ms/paragraph (median), batch 1 and batch 16, max_len 512.", "",
         "| model | passes | code runs | cost | test acc | macro-F1 | governance recall | params | non-emb | layers/FFN | MB | ms b1 | ms b16 | target |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    pm = f"{r['params'] / 1e6:.1f}M" if isinstance(r["params"], int) else "-"
    ne = f"{r['nonemb'] / 1e6:.1f}M" if isinstance(r["nonemb"], int) else "-"
    lines.append(f"| {r['name']} | {r['passes']} | {r['code_runs']} | {r['cost']} | {r['acc']:.4f} | {r['f1']:.4f} | {r['gov']:.3f} | {pm} | {ne} | {r['shape']} | {r['mb']} | {r['b1']} | {r['b16']} | {r['target']} |")
out = RUNS / "r1" / "compare.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwritten to {out}")

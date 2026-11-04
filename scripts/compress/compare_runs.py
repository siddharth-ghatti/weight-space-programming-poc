#!/usr/bin/env python
"""Render runs/c1/compare.md: pruning runs + compression baselines + reference rows."""
import json
from pathlib import Path
from wsp.model.weights import ROOT

RUNS = ROOT / "runs"
rows = []
paths = sorted(RUNS.glob("c1/*/eval/scores.json")) + sorted(RUNS.glob("_baselines/compress/*/eval/scores.json"))
for scores in paths:
    s = json.loads(scores.read_text())
    run = scores.parent.parent
    rel = run.relative_to(RUNS).as_posix()
    st = s.get("run_state", {})
    m = s.get("model", {})
    lat = s.get("latency", {}).get("ms_per_sentence", {})
    rows.append({
        "name": rel.replace("_baselines/compress/", "baseline:"),
        "passes": f"{st.get('passes_used', '-')}/{st.get('passes', '-')}" if st else "-",
        "code_runs": st.get("code_runs", "-") if st else "-",
        "cost": f"${st['cost_usd']:.2f}" if st else "-",
        "acc": s["test"]["accuracy"], "f1": s["test"]["macro_f1"],
        "params": m.get("total_params"), "nonemb": m.get("non_embedding_params"),
        "shape": f"{m.get('num_hidden_layers', '-')}L/{m.get('intermediate_size', '-')}" if m else "-",
        "mb": m.get("checkpoint_mb", "-"),
        "b1": lat.get("1", "-"), "b32": lat.get("32", "-"),
        "target": "-" if "meets_target" not in s else ("pass" if s["meets_target"] else "fail"),
    })
lines = ["# Pruning results (c1)", "",
         "Accuracy target: held-out test accuracy >= 97.8%. Params are the exported checkpoint; the 23.8M",
         "embedding parameters are the floor. Latency is CPU ms/sentence (median), batch 1 and batch 32.", "",
         "| model | passes | code runs | cost | test acc | macro-F1 | params | non-emb | layers/FFN | MB | ms b1 | ms b32 | target |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    pm = f"{r['params'] / 1e6:.1f}M" if isinstance(r["params"], int) else "-"
    ne = f"{r['nonemb'] / 1e6:.1f}M" if isinstance(r["nonemb"], int) else "-"
    lines.append(f"| {r['name']} | {r['passes']} | {r['code_runs']} | {r['cost']} | {r['acc']:.4f} | {r['f1']:.4f} | {pm} | {ne} | {r['shape']} | {r['mb']} | {r['b1']} | {r['b32']} | {r['target']} |")
out = RUNS / "c1" / "compare.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwritten to {out}")

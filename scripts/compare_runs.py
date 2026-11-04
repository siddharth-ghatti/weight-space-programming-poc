#!/usr/bin/env python
"""Render runs/compare.md: one table across all runs and baselines that have eval/scores.json."""
import json
from pathlib import Path
from wsp.model.weights import ROOT

rows = []
for scores in sorted((ROOT / "runs").glob("**/eval/scores.json")):
    s = json.loads(scores.read_text())
    run = scores.parent.parent
    rel = run.relative_to(ROOT / "runs").as_posix()
    name = rel if run.parent.name != "_baselines" else f"baseline:{run.name}"
    st = s.get("run_state", {})
    man = s.get("edit_manifest", {})
    rows.append({
        "name": name,
        "passes": f"{st.get('passes_used', '-')}/{st.get('passes', '-')}" if st else "-",
        "code_runs": st.get("code_runs", "-") if st else "-",
        "cost": f"${st['cost_usd']:.2f}" if st else "-",
        "acc": s["test"]["accuracy"], "f1": s["test"]["macro_f1"],
        "practice_acc": s.get("practice", {}).get("accuracy"),
        "edited": man.get("num_edited_tensors", "-"),
        "frac": man.get("fraction_of_model_changed"),
    })
lines = ["# Results", "", "| model | passes | code runs | cost | test acc | test macro-F1 | practice acc | tensors edited | fraction changed |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    pa = f"{r['practice_acc']:.4f}" if isinstance(r["practice_acc"], float) else "-"
    fr = f"{100 * r['frac']:.4f}%" if isinstance(r["frac"], float) else "-"
    lines.append(f"| {r['name']} | {r['passes']} | {r['code_runs']} | {r['cost']} | {r['acc']:.4f} | {r['f1']:.4f} | {pa} | {r['edited']} | {fr} |")
out = ROOT / "runs" / "compare.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwritten to {out}")

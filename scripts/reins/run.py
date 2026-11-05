#!/usr/bin/env python
"""Run the reinsurance pruning editor: uv run python scripts/reins/run.py --name r1/p25 --passes 25"""
import argparse
from pathlib import Path
from wspr.driver import Driver, load_config
from wspr.eval.evaluate import evaluate_run
from wspr.audit import audit_run

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--passes", type=int)
p.add_argument("--model", dest="editor_model")
p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
p.add_argument("--max-turns", dest="max_turns", type=int)
p.add_argument("--max-code-runs", dest="max_code_runs", type=int)
p.add_argument("--handoff-every", dest="handoff_every", type=int)
p.add_argument("--dollar-cap", dest="dollar_cap", type=float)
p.add_argument("--config", type=Path)
p.add_argument("--resume", action="store_true")
p.add_argument("--no-eval", action="store_true")
a = p.parse_args()
cfg = load_config(a.config, passes=a.passes, editor_model=a.editor_model, effort=a.effort, max_turns=a.max_turns,
                  max_code_runs=a.max_code_runs, handoff_every=a.handoff_every, dollar_cap=a.dollar_cap)
if cfg.get("accuracy_target") is None or cfg.get("macro_f1_target") is None:
    raise SystemExit("targets are not set; run scripts/reins/measure_start.py first")
d = Driver(a.name, cfg, resume=a.resume)
state = d.run()
print(f"run {a.name} ended: {state.end_reason}; passes {state.passes_used}/{state.passes}; code runs {state.code_runs}; ${state.cost_usd:.2f}")
if not a.no_eval:
    s = evaluate_run(d.run_dir)
    m = s["model"]
    print(f"TEST accuracy {s['test']['accuracy']:.4f}  macro-F1 {s['test']['macro_f1']:.4f}  params {m['total_params']:,} "
          f"({100 * m['reduction_vs_full']:.1f}% smaller; non-embedding {100 * m.get('non_embedding_reduction_vs_full', 0):.1f}%)  "
          f"target {'PASS' if s.get('meets_target') else 'FAIL'}")
    audit_run(d.run_dir)
    print(f"see {d.run_dir}/eval/report.md and {d.run_dir}/audit.json")

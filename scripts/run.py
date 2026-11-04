#!/usr/bin/env python
"""Run the editor: uv run python scripts/run.py --name p10 --passes 10"""
import argparse
from pathlib import Path
from wsp.harness.driver import Driver, load_config
from wsp.eval.evaluate import evaluate_run
from wsp.harness.audit import audit_run

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
p.add_argument("--no-eval", action="store_true", help="skip automatic evaluation after the run")
a = p.parse_args()
cfg = load_config(a.config, passes=a.passes, editor_model=a.editor_model, effort=a.effort, max_turns=a.max_turns,
                  max_code_runs=a.max_code_runs, handoff_every=a.handoff_every, dollar_cap=a.dollar_cap)
d = Driver(a.name, cfg, resume=a.resume)
state = d.run()
print(f"run {a.name} ended: {state.end_reason}; passes {state.passes_used}/{state.passes}; code runs {state.code_runs}; ${state.cost_usd:.2f}")
if not a.no_eval:
    s = evaluate_run(d.run_dir)
    print(f"TEST accuracy {s['test']['accuracy']:.4f}  macro-F1 {s['test']['macro_f1']:.4f}")
    audit_run(d.run_dir)
    print(f"see {d.run_dir}/eval/report.md and {d.run_dir}/audit.json")

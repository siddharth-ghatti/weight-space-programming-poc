"""Post-hoc audit of a reinsurance pruning run: the training harness's checks plus the structure history."""

from __future__ import annotations

import json
import re
from pathlib import Path

from wsp.harness.audit import SELF_AUTHORED_THRESHOLD, count_sentence_literals, lint


def audit_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    calls = []
    tc = run_dir / "tool_calls.jsonl"
    if tc.exists():
        calls = [json.loads(l) for l in tc.read_text().splitlines() if l.strip()]
    code_calls = [c for c in calls if c.get("tool") == "run_python"]
    flagged = []
    for c in code_calls:
        rej, fl = lint(c["input"].get("code", ""))
        if rej or fl:
            flagged.append({"turn": c["turn"], "rejected": rej, "flags": fl, "snippet": c["input"].get("code", "")[:300]})
    heavy = []
    for c in code_calls:
        code = c["input"].get("code", "")
        n_layers = len(set(re.findall(r"\.encoder\.layer\.(\d+)\.", code)))
        if n_layers >= 4 and re.search(r"\b(erf|gelu|softmax)\b", code):
            heavy.append(c["turn"])
    self_authored = [c["turn"] for c in code_calls if count_sentence_literals(c["input"].get("code", "")) >= SELF_AUTHORED_THRESHOLD]
    texts_requests = [c["turn"] for c in calls if c.get("tool") == "forward_pass" and c["input"].get("texts")]
    history = []
    for c in code_calls:
        for m in re.finditer(r"structure: layers=(\[[^\]]*\]) m=(\d+) params=([\d,]+)", c.get("result", "")):
            history.append({"turn": c["turn"], "layers": json.loads(m.group(1)), "m": int(m.group(2)),
                            "params": int(m.group(3).replace(",", ""))})
    state = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else {}
    man_p = run_dir / "edit" / "manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else {}
    report = {
        "run": f"{run_dir.parent.name}/{run_dir.name}",
        "start_model": manifest.get("diff_relative_to"),
        "turns": state.get("turn"),
        "code_calls": len(code_calls),
        "code_runs": len([c for c in code_calls if not c.get("rejected")]),
        "forward_passes": state.get("passes_used"),
        "cost_usd": state.get("cost_usd"),
        "lint_rejections_total": state.get("lint_rejections_total"),
        "flagged_calls": flagged,
        "possible_in_kernel_forward_turns": heavy,
        "possible_self_authored_data_turns": self_authored,
        "forward_pass_calls_with_free_texts": texts_requests,
        "edited_tensors": [t["name"] for t in manifest.get("tensors", [])],
        "fraction_of_start_model_changed": manifest.get("fraction_of_model_changed"),
        "structure": manifest.get("structure"),
        "exported_params": manifest.get("exported_params"),
        "structure_invalid": manifest.get("structure_invalid"),
        "structure_history": history,
    }
    (run_dir / "audit.json").write_text(json.dumps(report, indent=2))
    return report

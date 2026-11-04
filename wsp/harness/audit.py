"""Static lint of editor code (per run) and post-hoc audit of a whole run directory.

Policy (same as the paper): no backprop / optimizers / iterative fitting; no running the model inside
the kernel. Hard rejections are the unambiguous cases; everything else is flagged for human review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REJECT_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*(import|from)\s+torch\b", "torch is not allowed in the sandbox"),
    (r"^\s*(import|from)\s+transformers\b", "transformers is not allowed in the sandbox"),
    (r"^\s*(import|from)\s+(jax|tensorflow|keras|autograd|flax|optax)\b", "autodiff frameworks are not allowed"),
    (r"^\s*(import|from)\s+sklearn\b", "sklearn is not allowed (no fitted estimators)"),
    (r"\bscipy\.optimize\b|from\s+scipy\s+import\s+optimize|from\s+scipy\.optimize", "scipy.optimize is iterative fitting and is not allowed"),
    (r"^\s*(import|from)\s+(subprocess|socket|urllib|requests|httpx|http)\b|\b(subprocess|socket|urllib|requests|httpx)\.\w+\(", "no subprocesses or network"),
]

# Flagged (not rejected) for the post-hoc audit.
FLAG_PATTERNS: list[tuple[str, str]] = [
    (r"\b(learning_rate|lr\s*=|optimizer|epoch|gradient|\.grad\b|backward)\b", "training-loop vocabulary"),
    (r"\blstsq\b|\bpinv\b|\bsolve\(", "closed-form fit (allowed, review that it is not iterated)"),
    (r"\b(erf|gelu|softmax)\b", "activation function: check it is not a reimplemented forward pass"),
    (r"for\s+\w+\s+in\s+range\(\s*(1[2-9]|[2-9]\d+)\s*\)", "long loop: check for iterative optimization"),
]


def count_sentence_literals(code: str, min_chars: int = 50) -> int:
    """Rough count of prose-like string literals (lines inside triple quotes, or quoted strings) of
    at least min_chars that look like sentences. Used to flag possible self-authored data."""
    n = 0
    for block in re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', code, flags=re.S):
        for line in "".join(block).splitlines():
            if len(line.strip()) >= min_chars and len(line.split()) >= 8:
                n += 1
    stripped = re.sub(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', "", code, flags=re.S)
    for lit in re.findall(r'"([^"\n]{%d,})"|\'([^\'\n]{%d,})\'' % (min_chars, min_chars), stripped):
        t = "".join(lit)
        if len(t.split()) >= 8:
            n += 1
    return n


SELF_AUTHORED_THRESHOLD = 8


def lint(code: str) -> tuple[list[str], list[str]]:
    """Return (rejections, flags)."""
    rejections, flags = [], []
    if count_sentence_literals(code) >= SELF_AUTHORED_THRESHOLD:
        flags.append("possible self-authored sentences (many prose string literals)")
    for pat, msg in REJECT_PATTERNS:
        if re.search(pat, code, flags=re.MULTILINE):
            rejections.append(msg)
    for pat, msg in FLAG_PATTERNS:
        if re.search(pat, code, flags=re.MULTILINE):
            flags.append(msg)
    return rejections, flags


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
    # count encoder-layer references per run to spot in-kernel forward reimplementation
    heavy = []
    for c in code_calls:
        code = c["input"].get("code", "")
        n_layers = len(set(re.findall(r"bert\.encoder\.layer\.(\d+)\.", code)))
        if n_layers >= 6 and re.search(r"\b(erf|gelu|softmax)\b", code):
            heavy.append(c["turn"])
    self_authored = [c["turn"] for c in code_calls if count_sentence_literals(c["input"].get("code", "")) >= SELF_AUTHORED_THRESHOLD]
    texts_requests = [c["turn"] for c in calls if c.get("tool") == "forward_pass" and c["input"].get("texts")]
    state = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else {}
    manifest = json.loads((run_dir / "edit" / "manifest.json").read_text()) if (run_dir / "edit" / "manifest.json").exists() else {}
    report = {
        "run": run_dir.name,
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
        "fraction_of_model_changed": manifest.get("fraction_of_model_changed"),
    }
    (run_dir / "audit.json").write_text(json.dumps(report, indent=2))
    return report

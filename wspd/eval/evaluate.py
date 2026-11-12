"""Score an arm on the held-out test set: generated first word and scored-at-answer-position metrics, calibration,
selective accuracy; plus the forgetting suite. Every arm (unedited, teacher, direct edit, LoRA, ablations) goes
through the same code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wsp.eval.evaluate import full_metrics
from wsp.labels import LABELS
from wspd.common import K, ece, selective_accuracy


def _fill_unanswered(gold: list[int], pred: list[int | None]) -> list[int]:
    """An unanswered row counts as a wrong prediction (a label that is not the gold one)."""
    return [p if p is not None else (g + 1) % K for g, p in zip(gold, pred)]


def score_generation(gold: list[int], gen: dict) -> dict:
    pred = _fill_unanswered(gold, gen["pred"])
    m = full_metrics(gold, pred)
    m["answered"] = gen["answered"]; m["agree_teacher"] = gen["agree"]
    return m


@torch.no_grad()
def score_at_answer_position(bench, split: str = "test") -> dict:
    _, T, y = bench.data[split]
    lp, p, mass = bench.option_probs(split)
    pred = p.argmax(-1)
    m = full_metrics(y.tolist(), pred.tolist())
    m["ece"] = ece(p, y); m["selective"] = selective_accuracy(p, y)
    m["agree_teacher"] = (pred == T.argmax(-1)).float().mean().item()
    m["mass_on_options"] = mass.mean().item()
    m["teacher_accuracy"] = (T.argmax(-1) == y).float().mean().item()
    return m


def render_report(s: dict) -> str:
    g, sc = s.get("generation", {}), s.get("scored", {})
    ret = s.get("retention", {}); ret0 = s.get("retention_before", {})
    lines = [f"# {s.get('name', '')}", "",
             f"Test set: {sc.get('n', g.get('n', '-'))} held-out sentences. Teacher accuracy {100 * sc.get('teacher_accuracy', 0):.2f}%.", "",
             "| metric | generated first word | scored at the answer position |", "|---|---|---|",
             f"| accuracy | {100 * g.get('accuracy', 0):.2f}% | {100 * sc.get('accuracy', 0):.2f}% |",
             f"| macro-F1 | {g.get('macro_f1', 0):.4f} | {sc.get('macro_f1', 0):.4f} |",
             f"| agreement with the teacher | {100 * g.get('agree_teacher', 0):.1f}% | {100 * sc.get('agree_teacher', 0):.1f}% |",
             f"| answered with an option word | {100 * g.get('answered', 0):.1f}% | mass on options {sc.get('mass_on_options', 0):.3f} |",
             f"| ECE | - | {sc.get('ece', 0):.3f} |",
             f"| selective accuracy @80% coverage | - | {100 * sc.get('selective', {}).get('accuracy', 0):.2f}% |", ""]
    if ret:
        lines += ["| forgetting | before | after |", "|---|---|---|",
                  f"| WikiText-2 perplexity | {ret0.get('wiki_ppl', 0):.2f} | {ret.get('wiki_ppl', 0):.2f} |",
                  f"| WikiText-2 KL (nats/token) | 0 | {ret.get('wiki_kl', 0):.4f} |",
                  f"| ARC-Easy ({ret.get('arc_n', 0)}) | {100 * ret0.get('arc_acc', 0):.1f}% | {100 * ret.get('arc_acc', 0):.1f}% |",
                  f"| capitals | {100 * ret0.get('capitals_acc', 0):.1f}% | {100 * ret.get('capitals_acc', 0):.1f}% |",
                  f"| free text contaminated | {100 * ret0.get('contamination', 0):.0f}% | {100 * ret.get('contamination', 0):.0f}% |", ""]
        if s.get("forgot") is not None:
            lines.append("Forgetting verdict: " + ("; ".join(s["forgot"]) if s["forgot"] else "no sign of catastrophic forgetting") + "\n")
    if s.get("edit"):
        e = s["edit"]
        lines += [f"Edit: layer {e.get('layer')}, {e.get('n_columns')} columns, lambda {e.get('lambda')}, best round {e.get('best_round')}, "
                  f"{e.get('weights_changed', 0):,} weights changed, solve+write {e.get('edit_ms', 0):.1f} ms in all, wall {e.get('wall_seconds', 0):.0f} s.", ""]
    if s.get("per_class_text"):
        lines.append(s["per_class_text"])
    return "\n".join(lines)


def write_scores(run_dir: Path, scores: dict) -> None:
    run_dir = Path(run_dir); (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    sc = scores.get("scored", {}).get("per_class")
    if sc:
        rows = ["| label | precision | recall | F1 | n |", "|---|---|---|---|---|"]
        rows += [f"| {l} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |" for l, v in sc.items()]
        scores["per_class_text"] = "Per class (scored at the answer position):\n\n" + "\n".join(rows) + "\n"
    (run_dir / "eval" / "scores.json").write_text(json.dumps(scores, indent=1, default=float))
    (run_dir / "eval" / "report.md").write_text(render_report(scores))

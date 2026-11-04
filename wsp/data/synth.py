"""Synthesize the labeled insurance-sentence dataset with Claude, dedupe, and split.

Output: data/practice.jsonl (visible to the editor), data/test.jsonl (held out), data/stats.json.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from wsp.labels import DESCRIPTIONS, LABELS
from wsp.model.weights import ROOT

DATA = ROOT / "data"

LINES = [
    "personal auto", "homeowners", "individual health", "term life", "commercial general liability",
    "travel", "workers' compensation",
]
REGISTERS = {
    "policy wording": "formal policy/contract language as printed in the policy document",
    "broker summary": "an agent or broker summarizing the policy for a customer in plain English",
    "claims letter": "correspondence from the insurer's claims department to a policyholder",
    "customer FAQ": "question-and-answer style help-center text written for policyholders",
}
VARIANTS = {
    "concise": "Keep sentences short (8-18 words).",
    "standard": "Mix sentence lengths (12-30 words).",
    "verbose": "Use longer, more detailed sentences (25-45 words) with specific amounts, dates or named parties.",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS},
                },
                "required": ["text", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sentences"],
    "additionalProperties": False,
}


def prompt_for(line: str, register: str, variant: str, per_label: int) -> str:
    labels = "\n".join(f"- `{l}`: {DESCRIPTIONS[l]}" for l in LABELS)
    return f"""Write {per_label} sentences for EACH of the {len(LABELS)} labels below ({per_label * len(LABELS)} total), for a
**{line}** insurance policy, in the register of **{register}** ({REGISTERS[register]}).
{VARIANTS[variant]}

Labels:
{labels}

Requirements:
- Each sentence must belong unambiguously to exactly one label. Avoid sentences that mix two
  categories (e.g. a deductible mentioned inside a claims-procedure sentence).
- Each sentence must be self-contained and understandable alone (no dangling "it", "this section").
- Vary vocabulary, structure and specifics: invent realistic insurer names, amounts, percentages,
  time limits, perils, and defined terms appropriate to {line} insurance. Do not reuse the same
  opening words across sentences.
- Definitions should read like policy definitions ("'Occurrence' means ...") or FAQ explanations of a
  term, as fits the register.
- Do not number the sentences and do not include the label name inside the sentence text.
"""


def generate_batch(client: anthropic.Anthropic, model: str, line: str, register: str, variant: str, per_label: int) -> list[dict]:
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt_for(line, register, variant, per_label)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {getattr(resp, 'stop_details', None)}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    out = []
    for s in data["sentences"]:
        if s["label"] in LABELS and s["text"].strip():
            out.append({"text": s["text"].strip(), "label": s["label"], "line": line, "register": register, "variant": variant})
    return out


def normalize(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def tokens(t: str) -> set[str]:
    return set(normalize(t).split())


def dedupe(rows: list[dict], jaccard: float = 0.8) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    kept_toks: list[set[str]] = []
    for r in rows:
        n = normalize(r["text"])
        if n in seen:
            continue
        tk = tokens(r["text"])
        dup = False
        for kt in kept_toks:
            inter = len(tk & kt)
            if inter and inter / len(tk | kt) >= jaccard:
                dup = True
                break
        if dup:
            continue
        seen.add(n)
        kept.append(r)
        kept_toks.append(tk)
    return kept


def split(rows: list[dict], n_practice: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {l: [] for l in LABELS}
    for r in rows:
        by_label[r["label"]].append(r)
    practice, test = [], []
    per_label = n_practice // len(LABELS)
    for l in LABELS:
        items = by_label[l]
        rng.shuffle(items)
        practice += items[:per_label]
        test += items[per_label:]
    rng.shuffle(practice)
    rng.shuffle(test)
    return practice, test


def write_jsonl(rows: list[dict], path: Path, prefix: str) -> None:
    with path.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{prefix}{i:04d}", **r}) + "\n")


def build_dataset(model: str = "claude-opus-5-5", per_label: int = 5, workers: int = 4,
                  n_practice: int = 1500, seed: int = 0, raw_cache: Path | None = None) -> dict:
    DATA.mkdir(exist_ok=True)
    raw_cache = raw_cache or DATA / "raw_generated.jsonl"
    client = anthropic.Anthropic(max_retries=5)
    jobs = [(line, reg, var) for line in LINES for reg in REGISTERS for var in VARIANTS]
    rows: list[dict] = []
    done_keys: set[tuple] = set()
    if raw_cache.exists():  # resume support
        for l in raw_cache.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows.append(r)
                done_keys.add((r["line"], r["register"], r["variant"]))
    todo = [j for j in jobs if j not in done_keys]
    print(f"{len(jobs)} batches, {len(todo)} to generate ({len(rows)} rows cached)")
    with ThreadPoolExecutor(max_workers=workers) as ex, raw_cache.open("a") as f:
        futs = {ex.submit(generate_batch, client, model, *j, per_label): j for j in todo}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                batch = fut.result()
            except Exception as e:
                print(f"batch {j} failed: {e}")
                continue
            for r in batch:
                f.write(json.dumps(r) + "\n")
            f.flush()
            rows += batch
            print(f"batch {j}: {len(batch)} sentences (total {len(rows)})")
    rng = random.Random(seed)
    rng.shuffle(rows)
    kept = dedupe(rows)
    practice, test = split(kept, n_practice, seed)
    write_jsonl(practice, DATA / "practice.jsonl", "p")
    write_jsonl(test, DATA / "test.jsonl", "t")
    stats = {
        "generator_model": model,
        "raw": len(rows), "after_dedupe": len(kept),
        "practice": len(practice), "test": len(test),
        "practice_labels": dict(Counter(r["label"] for r in practice)),
        "test_labels": dict(Counter(r["label"] for r in test)),
        "lines": dict(Counter(r["line"] for r in kept)),
        "registers": dict(Counter(r["register"] for r in kept)),
        "test_sha256": hashlib.sha256((DATA / "test.jsonl").read_bytes()).hexdigest(),
        "practice_sha256": hashlib.sha256((DATA / "practice.jsonl").read_bytes()).hexdigest(),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats

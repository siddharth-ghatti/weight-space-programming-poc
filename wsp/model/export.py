"""Export base + edits as a standard, runnable Hugging Face checkpoint directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from wsp.model import weights as W
from wsp.model.forward import bert_config


def export_checkpoint(weights: dict[str, np.ndarray], out_dir: Path, manifest: dict | None = None) -> Path:
    """Write model.safetensors + config.json + tokenizer files to out_dir.

    Loads with `AutoModelForSequenceClassification.from_pretrained(out_dir)`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bert_config()
    cfg.architectures = ["BertForSequenceClassification"]
    cfg.save_pretrained(str(out_dir))
    state = W.model_state(weights)
    W.save_safetensors(state, out_dir / "model.safetensors")
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        src = W.BASE_DIR / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)
    # make tokenizer_config complete enough for AutoTokenizer
    tc_path = out_dir / "tokenizer_config.json"
    tc = json.loads(tc_path.read_text()) if tc_path.exists() else {}
    tc.setdefault("tokenizer_class", "BertTokenizer")
    tc.setdefault("do_lower_case", True)
    tc.setdefault("model_max_length", 512)
    tc_path.write_text(json.dumps(tc, indent=2))
    if manifest is not None:
        (out_dir / "edit_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_model_card(out_dir, manifest)
    return out_dir


def write_model_card(out_dir: Path, manifest: dict | None, extra: str = "") -> None:
    """A minimal Hugging Face model card so the folder can be pushed with scripts/push_hf.py as is."""
    from wsp.labels import LABELS
    man = manifest or {}
    tensors = man.get("tensors", [])
    tl = "\n".join(f"- `{t['name']}` ({t['changed_entries']:,} entries changed)" for t in tensors) or "- (none)"
    card = f"""---
base_model: bert-base-uncased
language: en
license: apache-2.0
pipeline_tag: text-classification
tags:
- weight-space-programming
- insurance
- bert
---

# Insurance sentence classifier built by weight editing (no gradient descent)

`BertForSequenceClassification` over `bert-base-uncased`, turned into an 8-label insurance sentence
classifier by an editor LLM that read and wrote the weights directly (closed-form math on weights and
on activations from a budgeted number of forward passes; no backprop, no optimizers). See the project
README for the protocol and results.

Labels: {", ".join(f"`{l}`" for l in LABELS)}

Edited tensors ({len(tensors)}, {100 * man.get("fraction_of_model_changed", 0):.3f}% of parameters changed):
{tl}

`edit_manifest.json` in this folder holds the exact diff metadata (sha256 of base and edited tensors).

```python
from transformers import pipeline
clf = pipeline("text-classification", model="<this repo>")
clf("Flood damage is not covered under this policy.")
```
{extra}"""
    (out_dir / "README.md").write_text(card)


def export_from_edit_dir(edit_dir: Path, out_dir: Path) -> Path:
    base = W.load_base()
    edits = W.load_edit_bundle(edit_dir, base, verify=True)
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    return export_checkpoint(W.apply_edits(base, edits), out_dir, manifest)

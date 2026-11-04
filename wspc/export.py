"""Export a pruned checkpoint that loads with stock transformers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from transformers import BertForSequenceClassification

from wsp.labels import LABELS
from wsp.model import weights as W
from wspc.forward import bert_config
from wspc.structure import FULL_PARAMS, Structure, count_params, prune_weights, structure_from_weights


def export_checkpoint(weights: dict[str, np.ndarray], out_dir: Path, manifest: dict | None = None, *,
                      structure: Structure | None = None, prune: bool = True, card_extra: str = "") -> Path:
    """Write model.safetensors + config.json + tokenizer files + structure/param info to out_dir.

    prune=True: `weights` are full-shaped and `structure` is applied here. prune=False: `weights` are
    already pruned (gradient-trained baselines); `structure` is only recorded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prune and structure is not None and not structure.is_full():
        weights = prune_weights(weights, structure)
    state = W.model_state(weights)
    k, m = structure_from_weights(state)
    cfg = bert_config(k, m)
    cfg.architectures = ["BertForSequenceClassification"]
    ref = BertForSequenceClassification(cfg).state_dict()
    ref = {kk: tuple(v.shape) for kk, v in ref.items() if "position_ids" not in kk}
    got = {kk: tuple(v.shape) for kk, v in state.items()}
    if got != ref:
        extra = sorted(set(got) - set(ref))[:5]; missing = sorted(set(ref) - set(got))[:5]
        bad = [kk for kk in got if kk in ref and got[kk] != ref[kk]][:5]
        raise RuntimeError(f"pruned state does not match config k={k} m={m}: extra={extra} missing={missing} shape_mismatch={bad}")
    cfg.save_pretrained(str(out_dir))
    W.save_safetensors(state, out_dir / "model.safetensors")
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        src = W.BASE_DIR / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)
    tc_path = out_dir / "tokenizer_config.json"
    tc = json.loads(tc_path.read_text()) if tc_path.exists() else {}
    tc.setdefault("tokenizer_class", "BertTokenizer")
    tc.setdefault("do_lower_case", True)
    tc.setdefault("model_max_length", 512)
    tc_path.write_text(json.dumps(tc, indent=2))
    pc = count_params(state)
    pc["checkpoint_bytes"] = (out_dir / "model.safetensors").stat().st_size
    (out_dir / "param_count.json").write_text(json.dumps(pc, indent=2))
    if structure is not None:
        (out_dir / "structure.json").write_text(json.dumps(structure.to_dict(), indent=1))
    if manifest is not None:
        (out_dir / "edit_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_model_card(out_dir, manifest, pc, card_extra)
    return out_dir


def write_model_card(out_dir: Path, manifest: dict | None, pc: dict | None = None, extra: str = "") -> None:
    out_dir = Path(out_dir)
    man = manifest or {}
    pc = pc or (json.loads((out_dir / "param_count.json").read_text()) if (out_dir / "param_count.json").exists() else {})
    st = man.get("structure") or {}
    init_from = man.get("init_from")
    total = pc.get("total", 0)
    reduction = 100 * (1 - total / FULL_PARAMS) if total else 0.0
    layers = st.get("layers", list(range(pc.get("num_layers", 12))))
    lines = [
        "---", "base_model: bert-base-uncased", "language: en", "license: apache-2.0", "pipeline_tag: text-classification",
        "tags:", "- weight-space-programming", "- pruning", "- insurance", "- bert", "---", "",
        "# Pruned insurance sentence classifier built by weight editing (no gradient descent)", "",
        "A smaller `BertForSequenceClassification` derived from `bert-base-uncased`. " + (
            f"Starting model: the weight-edited classifier from `{init_from}` (see the project README). "
            if init_from else "") +
        "An editor LLM chose which encoder layers and feed-forward units to keep and re-fit the remaining weights "
        "in closed form from a budgeted number of forward passes; no backprop, no optimizers, no extra data.", "",
        f"Labels: {', '.join(f'`{l}`' for l in LABELS)}", "",
        "| | value |", "|---|---|",
        f"| encoder layers kept (original ids) | {len(layers)}: {layers} |",
        f"| intermediate (FFN) size | {pc.get('intermediate_size', '-')} |",
        f"| total parameters | {total:,} |",
        f"| non-embedding parameters | {pc.get('non_embedding', 0):,} |",
        f"| reduction vs bert-base (109,488,392) | {reduction:.1f}% |",
        f"| checkpoint size | {pc.get('checkpoint_bytes', 0) / 1e6:.0f} MB |", "",
        "`edit_manifest.json`, `structure.json` and `param_count.json` in this folder record the exact diff "
        "against the starting model, the pruning structure and the parameter accounting.", "",
        "```python", "from transformers import pipeline",
        'clf = pipeline("text-classification", model="<this repo>")',
        'clf("Flood damage is not covered under this policy.")', "```", "",
    ]
    if man.get("note"):
        lines.insert(13, f"> {man['note']}\n")
    (out_dir / "README.md").write_text("\n".join(lines) + (extra or ""))


def export_from_edit_dir(edit_dir: Path, out_dir: Path) -> Path:
    """Rebuild bert-base + init_from edit + this edit + structure -> pruned checkpoint."""
    from wspc.bundle import load_bundle_structure, load_edit_bundle
    from wspc.start import load_start

    edit_dir = Path(edit_dir)
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    start, _ = load_start(manifest.get("init_from"))
    edits = load_edit_bundle(edit_dir, start, verify=True)
    structure = load_bundle_structure(edit_dir)
    return export_checkpoint(W.apply_edits(start, edits), out_dir, manifest, structure=structure)

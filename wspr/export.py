"""Export a pruned checkpoint that loads with stock transformers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from transformers import AutoModelForSequenceClassification

from wspr.arch import TOKENIZER_FILES, Arch
from wspr.forward import model_config
from wspr.structure import Structure, count_params, prune_weights, structure_from_weights


def export_checkpoint(weights: dict[str, np.ndarray], out_dir: Path, manifest: dict | None, arch: Arch, *,
                      structure: Structure | None = None, prune: bool = True, card_extra: str = "") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if prune and structure is not None and not structure.is_full(arch):
        weights = prune_weights(weights, structure, arch)
    state = {k: v for k, v in weights.items() if not arch.is_buffer(k)}
    k, m = structure_from_weights(state, arch)
    cfg = model_config(arch, k, m)
    ref = AutoModelForSequenceClassification.from_config(cfg).state_dict()
    ref = {kk: tuple(v.shape) for kk, v in ref.items() if "position_ids" not in kk}
    got = {kk: tuple(v.shape) for kk, v in state.items()}
    if got != ref:
        extra = sorted(set(got) - set(ref))[:5]; missing = sorted(set(ref) - set(got))[:5]
        bad = [kk for kk in got if kk in ref and got[kk] != ref[kk]][:5]
        raise RuntimeError(f"pruned state does not match config k={k} m={m}: extra={extra} missing={missing} shape_mismatch={bad}")
    cfg.save_pretrained(str(out_dir))
    save_file({kk: np.ascontiguousarray(v) for kk, v in state.items()}, str(out_dir / "model.safetensors"))
    for fn in TOKENIZER_FILES:
        src = arch.model_dir / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)
    pc = count_params(state, arch)
    pc["checkpoint_bytes"] = (out_dir / "model.safetensors").stat().st_size
    (out_dir / "param_count.json").write_text(json.dumps(pc, indent=2))
    if structure is not None:
        (out_dir / "structure.json").write_text(json.dumps(structure.to_dict(), indent=1))
    if manifest is not None:
        (out_dir / "edit_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_model_card(out_dir, manifest, arch, pc, card_extra)
    return out_dir


def write_model_card(out_dir: Path, manifest: dict | None, arch: Arch, pc: dict | None = None, extra: str = "") -> None:
    out_dir = Path(out_dir)
    man = manifest or {}
    pc = pc or (json.loads((out_dir / "param_count.json").read_text()) if (out_dir / "param_count.json").exists() else {})
    st = man.get("structure") or {}
    total = pc.get("total", 0)
    reduction = 100 * (1 - total / arch.full_params) if total else 0.0
    ne_red = 100 * pc.get("non_embedding_reduction_vs_full", 0.0)
    layers = st.get("layers", list(range(pc.get("num_layers", arch.num_layers))))
    repo = man.get("hub_repo") or man.get("base_model") or arch.model_dir.name
    commit = man.get("hub_commit")
    lines = [
        "---", f"base_model: {repo}", "language: en", "license: apache-2.0", "pipeline_tag: text-classification",
        "tags:", "- weight-space-programming", "- pruning", "- climate-risk", "- tcfd", "- reinsurance", "- roberta", "---", "",
        "# Pruned TCFD disclosure classifier built by weight editing (no gradient descent)", "",
        f"A smaller `RobertaForSequenceClassification` derived from [{repo}](https://huggingface.co/{repo})"
        + (f" (commit `{commit[:12]}`)" if commit else "") + ", ClimateBERT's classifier of climate-related "
        "disclosure paragraphs into TCFD recommendation categories. An editor LLM chose which encoder layers and "
        "feed-forward units to keep and re-fit the remaining weights in closed form from a budgeted number of "
        "forward passes on the model's public training split; no backprop, no optimizers, no extra data.", "",
        f"Labels: {', '.join(f'`{l}`' for l in arch.labels)}", "",
        "| | value |", "|---|---|",
        f"| encoder layers kept (original ids) | {len(layers)}: {layers} |",
        f"| intermediate (FFN) size | {pc.get('intermediate_size', '-')} |",
        f"| total parameters | {total:,} |",
        f"| non-embedding parameters | {pc.get('non_embedding', 0):,} |",
        f"| reduction vs the start model ({arch.full_params:,}) | {reduction:.1f}% total, {ne_red:.1f}% non-embedding |",
        f"| checkpoint size | {pc.get('checkpoint_bytes', 0) / 1e6:.0f} MB |", "",
        "`edit_manifest.json`, `structure.json` and `param_count.json` in this folder record the exact diff "
        "against the starting checkpoint, the pruning structure and the parameter accounting.", "",
        "```python", "from transformers import pipeline",
        'clf = pipeline("text-classification", model="<this repo>")',
        'clf("The board\'s risk committee reviews climate-related risks twice a year.")', "```", "",
        "Training data of the start model and practice data of the editor: `climatebert/tcfd_recommendations` "
        "(CC BY-NC-SA 4.0, research use). The model was never trained on the test split used for the accuracy line below.", "",
    ]
    if man.get("note"):
        lines.insert(15, f"> {man['note']}\n")
    (out_dir / "README.md").write_text("\n".join(lines) + (extra or ""))


def export_from_edit_dir(edit_dir: Path, out_dir: Path) -> Path:
    """Rebuild start checkpoint + this edit + structure -> pruned checkpoint."""
    from wspr.bundle import apply_edits, load_bundle_structure, load_edit_bundle
    from wspr.start import load_start

    edit_dir = Path(edit_dir)
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    arch, start, _ = load_start(manifest["start_model_dir"])
    edits = load_edit_bundle(edit_dir, start, verify=True)
    structure = load_bundle_structure(edit_dir, arch)
    return export_checkpoint(apply_edits(start, edits, arch), out_dir, manifest, arch, structure=structure)

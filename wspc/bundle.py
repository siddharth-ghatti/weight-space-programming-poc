"""Edit bundles for pruning runs: an exact diff against the start model plus the structure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wsp.model import weights as W
from wspc.structure import Structure, param_count, count_params, validate_structure, default_structure


def edit_manifest(start: dict[str, np.ndarray], edits: dict[str, np.ndarray], provenance: dict,
                  structure: Structure) -> dict:
    tensors = []
    total_changed = 0
    total_params = sum(int(np.prod(v.shape)) for k, v in start.items() if not W.is_extra(k))
    for name in sorted(edits):
        e, b = edits[name], start[name]
        changed = int(np.sum(e != b))
        total_changed += changed
        tensors.append({
            "name": name, "shape": list(e.shape), "changed_entries": changed,
            "fraction_changed": changed / max(1, e.size),
            "frobenius_delta": float(np.linalg.norm(e - b)), "frobenius_base": float(np.linalg.norm(b)),
            "sha256_base": W.sha256(b), "sha256_edited": W.sha256(e),
        })
    pc = param_count(structure)
    return {
        "base_model": provenance.get("base_model", "bert-base-uncased"),
        "diff_relative_to": provenance.get("init_from") or "bert-base-uncased",
        **{k: v for k, v in provenance.items() if k != "base_model"},
        "num_edited_tensors": len(tensors),
        "total_changed_entries": total_changed,
        "total_model_params": total_params,          # of the (unpruned) start model
        "fraction_of_model_changed": total_changed / max(1, total_params),
        "structure": structure.to_dict(),
        "param_count": pc,
        "exported_params": pc["total"],
        "tensors": tensors,
    }


def save_edit_bundle(start, edits, out_dir: Path, provenance: dict, structure: Structure) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = edit_manifest(start, edits, provenance, structure)
    np.savez_compressed(out_dir / "edit.npz", **edits)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "structure.json").write_text(json.dumps(structure.to_dict(), indent=1))
    return manifest


def load_edit_bundle(edit_dir: Path, start, verify: bool = True) -> dict[str, np.ndarray]:
    return W.load_edit_bundle(Path(edit_dir), start, verify=verify)


def load_bundle_structure(edit_dir: Path) -> Structure:
    p = Path(edit_dir) / "structure.json"
    return validate_structure(json.loads(p.read_text())) if p.exists() else default_structure()


def full_change_manifest(weights: dict[str, np.ndarray], provenance: dict, structure: Structure | None) -> dict:
    """For gradient-trained baselines whose tensors are already pruned: no per-tensor diff exists."""
    pc = count_params(weights)
    return {
        "base_model": provenance.get("base_model", "bert-base-uncased"),
        "diff_relative_to": None,
        **{k: v for k, v in provenance.items() if k != "base_model"},
        "note": "all parameters re-fit by gradient descent; not an edit bundle",
        "num_edited_tensors": len([k for k in weights if not W.is_extra(k)]),
        "total_changed_entries": pc["total"],
        "total_model_params": pc["total"],
        "fraction_of_model_changed": 1.0,
        "structure": structure.to_dict() if structure else None,
        "param_count": pc,
        "exported_params": pc["total"],
        "tensors": [],
    }

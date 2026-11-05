"""Edit bundles: an exact diff against the start checkpoint plus the pruning structure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wspr.arch import Arch
from wspr.start import tensor_sha256
from wspr.structure import Structure, count_params, default_structure, param_count, validate_structure


class EditError(ValueError):
    pass


def validate_edit(name: str, arr: np.ndarray, start: dict[str, np.ndarray], arch: Arch) -> np.ndarray:
    if name not in start:
        raise EditError(f"unknown tensor {name!r}")
    if arch.is_frozen(name):
        raise EditError(f"{name!r} is frozen (embeddings cannot be edited)")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != start[name].shape:
        raise EditError(f"shape mismatch for {name!r}: got {arr.shape}, expected {start[name].shape}")
    if not np.all(np.isfinite(arr)):
        raise EditError(f"{name!r} contains non-finite values")
    return np.ascontiguousarray(arr)


def load_edits(edits_dir: Path) -> dict[str, np.ndarray]:
    edits: dict[str, np.ndarray] = {}
    if not Path(edits_dir).exists():
        return edits
    for p in sorted(Path(edits_dir).glob("*.npy")):
        edits[p.stem] = np.load(p).astype(np.float32)
    return edits


def apply_edits(start: dict[str, np.ndarray], edits: dict[str, np.ndarray], arch: Arch) -> dict[str, np.ndarray]:
    out = dict(start)
    for k, v in edits.items():
        out[k] = validate_edit(k, v, start, arch)
    return out


def edit_manifest(start: dict[str, np.ndarray], edits: dict[str, np.ndarray], provenance: dict,
                  structure: Structure, arch: Arch) -> dict:
    tensors = []
    total_changed = 0
    total_params = sum(int(np.prod(v.shape)) for v in start.values())
    for name in sorted(edits):
        e, b = edits[name], start[name]
        changed = int(np.sum(e != b))
        total_changed += changed
        tensors.append({"name": name, "shape": list(e.shape), "changed_entries": changed,
                        "fraction_changed": changed / max(1, e.size),
                        "frobenius_delta": float(np.linalg.norm(e - b)), "frobenius_base": float(np.linalg.norm(b)),
                        "sha256_base": tensor_sha256(b), "sha256_edited": tensor_sha256(e)})
    pc = param_count(structure, arch)
    return {
        "base_model": provenance.get("base_model"),
        "diff_relative_to": f"{provenance.get('hub_repo')}@{provenance.get('hub_commit')}" if provenance.get("hub_repo") else provenance.get("start_model_dir"),
        **{k: v for k, v in provenance.items() if k != "base_model"},
        "num_edited_tensors": len(tensors),
        "total_changed_entries": total_changed,
        "total_model_params": total_params,
        "fraction_of_model_changed": total_changed / max(1, total_params),
        "structure": structure.to_dict(),
        "param_count": pc,
        "exported_params": pc["total"],
        "tensors": tensors,
    }


def save_edit_bundle(start, edits, out_dir: Path, provenance: dict, structure: Structure, arch: Arch) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = edit_manifest(start, edits, provenance, structure, arch)
    np.savez_compressed(out_dir / "edit.npz", **edits)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "structure.json").write_text(json.dumps(structure.to_dict(), indent=1))
    return manifest


def load_edit_bundle(edit_dir: Path, start: dict[str, np.ndarray], verify: bool = True) -> dict[str, np.ndarray]:
    npz = np.load(Path(edit_dir) / "edit.npz")
    edits = {k: npz[k].astype(np.float32) for k in npz.files}
    if verify:
        manifest = json.loads((Path(edit_dir) / "manifest.json").read_text())
        for t in manifest["tensors"]:
            n = t["name"]
            if tensor_sha256(start[n]) != t["sha256_base"]:
                raise EditError(f"start hash mismatch for {n}")
            if tensor_sha256(edits[n]) != t["sha256_edited"]:
                raise EditError(f"edited hash mismatch for {n}")
    return edits


def load_bundle_structure(edit_dir: Path, arch: Arch) -> Structure:
    p = Path(edit_dir) / "structure.json"
    return validate_structure(json.loads(p.read_text()), arch) if p.exists() else default_structure(arch)


def full_change_manifest(weights: dict[str, np.ndarray], provenance: dict, structure: Structure | None, arch: Arch) -> dict:
    pc = count_params(weights, arch)
    return {
        "base_model": provenance.get("base_model"),
        "diff_relative_to": None,
        **{k: v for k, v in provenance.items() if k != "base_model"},
        "note": "all parameters re-fit by gradient descent; not an edit bundle",
        "num_edited_tensors": len(weights),
        "total_changed_entries": pc["total"],
        "total_model_params": pc["total"],
        "fraction_of_model_changed": 1.0,
        "structure": structure.to_dict() if structure else None,
        "param_count": pc,
        "exported_params": pc["total"],
        "tensors": [],
    }

"""Load bert-base-uncased weights as numpy, define the editable/frozen sets, apply edits.

Tensor names are normalized to modern HF names (`LayerNorm.weight`/`.bias` rather than the
legacy `gamma`/`beta` found in the original checkpoint). Two tensors absent from the base are added,
zero-initialized: `classifier.weight` [NUM_LABELS, 768] and `classifier.bias` [NUM_LABELS].

The MLM/NSP head tensors (`cls.*`) are kept readable (they are useful for reading the model) but are
not part of the classification model and cannot be edited.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from wsp.labels import NUM_LABELS

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "models" / "bert-base-uncased"
HIDDEN = 768

_LEGACY = {"LayerNorm.gamma": "LayerNorm.weight", "LayerNorm.beta": "LayerNorm.bias"}


def normalize_name(name: str) -> str:
    for old, new in _LEGACY.items():
        if name.endswith(old):
            return name[: -len(old)] + new
    return name


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    editable: bool
    in_model: bool  # part of BertForSequenceClassification
    note: str = ""


def is_frozen(name: str) -> bool:
    return name.startswith("bert.embeddings.")


def is_extra(name: str) -> bool:
    return name.startswith("cls.")


def _added_tensors() -> dict[str, np.ndarray]:
    return {
        "classifier.weight": np.zeros((NUM_LABELS, HIDDEN), dtype=np.float32),
        "classifier.bias": np.zeros((NUM_LABELS,), dtype=np.float32),
    }


def load_base(base_dir: Path = BASE_DIR) -> dict[str, np.ndarray]:
    """All base tensors (normalized names, float32) + zero-initialized classifier head."""
    out: dict[str, np.ndarray] = {}
    with safe_open(str(base_dir / "model.safetensors"), "np") as f:
        for k in f.keys():
            out[normalize_name(k)] = np.ascontiguousarray(f.get_tensor(k).astype(np.float32))
    out.update(_added_tensors())
    return out


def catalog(weights: dict[str, np.ndarray] | None = None) -> list[TensorInfo]:
    weights = weights if weights is not None else load_base()
    infos = []
    for name in sorted(weights):
        w = weights[name]
        if is_extra(name):
            note = "MLM/NSP head from pretraining; readable, not editable, not in the classifier"
            infos.append(TensorInfo(name, w.shape, str(w.dtype), False, False, note))
        elif is_frozen(name):
            infos.append(TensorInfo(name, w.shape, str(w.dtype), False, True, "frozen"))
        elif name.startswith("classifier."):
            infos.append(TensorInfo(name, w.shape, str(w.dtype), True, True, "zero-initialized head"))
        else:
            infos.append(TensorInfo(name, w.shape, str(w.dtype), True, True, ""))
    return infos


def catalog_json(weights: dict[str, np.ndarray] | None = None) -> list[dict]:
    return [asdict(t) | {"shape": list(t.shape)} for t in catalog(weights)]


def model_state(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Subset that belongs to BertForSequenceClassification."""
    return {k: v for k, v in weights.items() if not is_extra(k)}


# ---------------------------------------------------------------- edits

class EditError(ValueError):
    pass


def validate_edit(name: str, arr: np.ndarray, base: dict[str, np.ndarray]) -> np.ndarray:
    if name not in base:
        raise EditError(f"unknown tensor {name!r}")
    if is_frozen(name):
        raise EditError(f"{name!r} is frozen (embeddings cannot be edited)")
    if is_extra(name):
        raise EditError(f"{name!r} is not part of the classification model and cannot be edited")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != base[name].shape:
        raise EditError(f"shape mismatch for {name!r}: got {arr.shape}, expected {base[name].shape}")
    if not np.all(np.isfinite(arr)):
        raise EditError(f"{name!r} contains non-finite values")
    return np.ascontiguousarray(arr)


def load_edits(edits_dir: Path) -> dict[str, np.ndarray]:
    """Read `/work/edits/<name>.npy` files written by the sandbox prelude."""
    edits: dict[str, np.ndarray] = {}
    if not edits_dir.exists():
        return edits
    for p in sorted(edits_dir.glob("*.npy")):
        edits[p.stem] = np.load(p).astype(np.float32)
    return edits


def apply_edits(base: dict[str, np.ndarray], edits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(base)
    for k, v in edits.items():
        out[k] = validate_edit(k, v, base)
    return out


def sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def edit_manifest(base: dict[str, np.ndarray], edits: dict[str, np.ndarray]) -> dict:
    tensors = []
    total_changed = 0
    total_params = sum(int(np.prod(v.shape)) for k, v in base.items() if not is_extra(k))
    for name in sorted(edits):
        e = edits[name]
        b = base[name]
        changed = int(np.sum(e != b))
        total_changed += changed
        delta = e - b
        tensors.append({
            "name": name,
            "shape": list(e.shape),
            "changed_entries": changed,
            "fraction_changed": changed / max(1, e.size),
            "frobenius_delta": float(np.linalg.norm(delta)),
            "frobenius_base": float(np.linalg.norm(b)),
            "sha256_base": sha256(b),
            "sha256_edited": sha256(e),
        })
    return {
        "base_model": "bert-base-uncased",
        "num_edited_tensors": len(tensors),
        "total_changed_entries": total_changed,
        "total_model_params": total_params,
        "fraction_of_model_changed": total_changed / max(1, total_params),
        "tensors": tensors,
    }


def save_edit_bundle(base: dict[str, np.ndarray], edits: dict[str, np.ndarray], out_dir: Path) -> dict:
    """Write edit.npz + manifest.json (the exact-diff form)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = edit_manifest(base, edits)
    np.savez_compressed(out_dir / "edit.npz", **edits)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_edit_bundle(edit_dir: Path, base: dict[str, np.ndarray] | None = None, verify: bool = True) -> dict[str, np.ndarray]:
    npz = np.load(edit_dir / "edit.npz")
    edits = {k: npz[k].astype(np.float32) for k in npz.files}
    if verify:
        base = base if base is not None else load_base()
        manifest = json.loads((edit_dir / "manifest.json").read_text())
        for t in manifest["tensors"]:
            n = t["name"]
            if sha256(base[n]) != t["sha256_base"]:
                raise EditError(f"base hash mismatch for {n}")
            if sha256(edits[n]) != t["sha256_edited"]:
                raise EditError(f"edited hash mismatch for {n}")
    return edits


def save_safetensors(weights: dict[str, np.ndarray], path: Path) -> None:
    save_file({k: np.ascontiguousarray(v) for k, v in weights.items()}, str(path))

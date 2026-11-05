"""The starting model: a third-party Hugging Face checkpoint downloaded by scripts/reins/fetch_model.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from wspr.arch import ROOT, Arch, load_arch

BASELINES_DIR = ROOT / "runs" / "_baselines" / "reins"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def load_checkpoint(model_dir: Path, arch: Arch | None = None) -> dict[str, np.ndarray]:
    """Float tensors of a checkpoint as float32 numpy (integer buffers such as position_ids are skipped)."""
    out: dict[str, np.ndarray] = {}
    with safe_open(str(Path(model_dir) / "model.safetensors"), "np") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if not np.issubdtype(t.dtype, np.floating):
                continue
            out[k] = np.ascontiguousarray(t.astype(np.float32))
    return out


def weights_hash(weights: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(weights):
        h.update(k.encode()); h.update(tensor_sha256(weights[k]).encode())
    return h.hexdigest()


def load_start(model_dir: Path | str) -> tuple[Arch, dict[str, np.ndarray], dict]:
    """(arch, weights, provenance) of the start checkpoint; weights are the exact model tensors."""
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    arch = load_arch(model_dir)
    weights = load_checkpoint(model_dir, arch)
    expected = set(_expected_keys(arch))
    got = set(weights)
    if got != expected:
        raise RuntimeError(f"checkpoint keys do not match the expected {arch.model_type} classifier: "
                           f"missing={sorted(expected - got)[:5]} unexpected={sorted(got - expected)[:5]}")
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    if total != arch.full_params:
        raise RuntimeError(f"parameter arithmetic {arch.full_params:,} != checkpoint numel {total:,}")
    prov_p = model_dir / "provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else {}
    try:
        shown = model_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        shown = model_dir.name
    provenance = {
        "base_model": prov.get("hub_repo", model_dir.name),
        "start_model_dir": shown,
        "hub_repo": prov.get("hub_repo"),
        "hub_commit": prov.get("hub_commit"),
        "start_safetensors_sha256": prov.get("safetensors_sha256") or file_sha256(model_dir / "model.safetensors"),
        "start_model_sha256": weights_hash(weights),
    }
    scores = BASELINES_DIR / "start" / "eval" / "scores.json"
    if scores.exists():
        s = json.loads(scores.read_text())
        provenance["start_test_accuracy"] = s["test"]["accuracy"]
        provenance["start_test_macro_f1"] = s["test"]["macro_f1"]
    return arch, weights, provenance


def _expected_keys(arch: Arch) -> list[str]:
    p = arch.model_prefix
    keys = [f"{p}.embeddings.word_embeddings.weight", f"{p}.embeddings.position_embeddings.weight",
            f"{p}.embeddings.token_type_embeddings.weight", f"{p}.embeddings.LayerNorm.weight", f"{p}.embeddings.LayerNorm.bias"]
    for l in range(arch.num_layers):
        lp = f"{arch.layer_prefix}{l}."
        for part in ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
                     "attention.output.LayerNorm", "intermediate.dense", "output.dense", "output.LayerNorm"):
            keys += [lp + part + ".weight", lp + part + ".bias"]
    keys += list(arch.head_keys)
    return keys

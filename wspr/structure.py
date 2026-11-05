"""Structural pruning spec parameterized by the start model's Arch: which encoder layers and which
feed-forward units survive. Edits are saved at full shape; the harness applies the structure when it
builds the edited model and when it exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wspr.arch import Arch


class StructureError(ValueError):
    pass


@dataclass(frozen=True)
class Structure:
    layers: tuple[int, ...]
    ffn_keep: dict[int, tuple[int, ...]]

    @property
    def m(self) -> int:
        return len(self.ffn_keep[self.layers[0]])

    @property
    def k(self) -> int:
        return len(self.layers)

    def is_full(self, arch: Arch) -> bool:
        full = tuple(range(arch.ffn))
        return self.layers == tuple(range(arch.num_layers)) and all(self.ffn_keep[l] == full for l in self.layers)

    def to_dict(self) -> dict:
        return {"version": 1, "layers": list(self.layers), "m": self.m,
                "ffn_keep": {str(l): list(self.ffn_keep[l]) for l in self.layers}}

    def layer_map(self) -> dict[int, int]:
        return {i + 1: l for i, l in enumerate(self.layers)}


def default_structure(arch: Arch) -> Structure:
    return Structure(tuple(range(arch.num_layers)), {l: tuple(range(arch.ffn)) for l in range(arch.num_layers)})


def validate_structure(d: dict, arch: Arch) -> Structure:
    L, F = arch.num_layers, arch.ffn
    if not isinstance(d, dict):
        raise StructureError("structure must be a JSON object")
    layers = d.get("layers")
    if not isinstance(layers, list) or not layers:
        raise StructureError("structure.layers must be a non-empty list of original layer ids")
    try:
        layers = [int(x) for x in layers]
    except (TypeError, ValueError):
        raise StructureError("structure.layers must contain integers")
    if any(l < 0 or l >= L for l in layers):
        raise StructureError(f"structure.layers must be within 0..{L - 1}")
    if any(b <= a for a, b in zip(layers, layers[1:])):
        raise StructureError("structure.layers must be strictly increasing (drop layers, do not reorder or repeat)")
    raw = d.get("ffn_keep")
    if raw is None:
        raw = {str(l): list(range(F)) for l in layers}
    if not isinstance(raw, dict):
        raise StructureError("structure.ffn_keep must be an object {layer: [unit ids]}")
    keep: dict[int, tuple[int, ...]] = {}
    for key, units in raw.items():
        try:
            l = int(key)
        except (TypeError, ValueError):
            raise StructureError(f"ffn_keep key {key!r} is not a layer id")
        if l not in layers:
            raise StructureError(f"ffn_keep has an entry for layer {l}, which is not kept")
        if not isinstance(units, list) or not units:
            raise StructureError(f"ffn_keep[{l}] must be a non-empty list of unit ids")
        try:
            u = [int(x) for x in units]
        except (TypeError, ValueError):
            raise StructureError(f"ffn_keep[{l}] must contain integers")
        if any(x < 0 or x >= F for x in u):
            raise StructureError(f"ffn_keep[{l}] unit ids must be within 0..{F - 1}")
        if len(set(u)) != len(u):
            raise StructureError(f"ffn_keep[{l}] has duplicate unit ids")
        if u != sorted(u):
            raise StructureError(f"ffn_keep[{l}] must be sorted ascending")
        keep[l] = tuple(u)
    missing = [l for l in layers if l not in keep]
    if missing:
        raise StructureError(f"ffn_keep is missing entries for kept layers {missing}")
    sizes = {len(keep[l]) for l in layers}
    if len(sizes) != 1:
        raise StructureError(f"all kept layers must keep the same number of FFN units (the checkpoint has one intermediate_size); got sizes {sorted(sizes)}")
    s = Structure(tuple(layers), keep)
    if "m" in d and d["m"] is not None and int(d["m"]) != s.m:
        raise StructureError(f"structure.m={d['m']} does not match the ffn_keep lists (length {s.m})")
    return s


def load_structure(work_dir: Path, arch: Arch) -> Structure:
    p = Path(work_dir) / "structure.json"
    if not p.exists():
        return default_structure(arch)
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise StructureError(f"structure.json is not valid JSON: {e}")
    return validate_structure(d, arch)


def structure_hash(s: Structure, arch: Arch) -> str:
    if s.is_full(arch):
        return "full"
    return hashlib.sha256(json.dumps(s.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def param_count(s: Structure, arch: Arch) -> dict:
    return arch.param_count(s.k, s.m)


def is_full_shaped(weights: dict[str, np.ndarray], arch: Arch) -> bool:
    lp = arch.layer_prefix
    if f"{lp}{arch.num_layers - 1}.attention.self.query.weight" not in weights:
        return False
    for l in range(arch.num_layers):
        w = weights.get(f"{lp}{l}.intermediate.dense.weight")
        if w is None or w.shape[0] != arch.ffn:
            return False
    return True


def prune_weights(weights: dict[str, np.ndarray], s: Structure, arch: Arch) -> dict[str, np.ndarray]:
    """Keep s.layers (renumbered 0..k-1) and, per kept layer, the FFN units in s.ffn_keep."""
    if not is_full_shaped(weights, arch):
        raise StructureError(f"prune_weights expects full-shaped ({arch.num_layers}-layer, {arch.ffn}-unit) weights; refusing to prune twice")
    lp = arch.layer_prefix
    out: dict[str, np.ndarray] = {k: v for k, v in weights.items() if not k.startswith(lp)}
    for new_i, old_i in enumerate(s.layers):
        old_p, new_p = f"{lp}{old_i}.", f"{lp}{new_i}."
        idx = np.asarray(s.ffn_keep[old_i], dtype=np.int64)
        for k, v in weights.items():
            if not k.startswith(old_p):
                continue
            suffix = k[len(old_p):]
            if suffix == "intermediate.dense.weight":
                v = v[idx, :]
            elif suffix == "intermediate.dense.bias":
                v = v[idx]
            elif suffix == "output.dense.weight":
                v = v[:, idx]
            out[new_p + suffix] = np.ascontiguousarray(v)
    return out


def structure_from_weights(weights: dict[str, np.ndarray], arch: Arch) -> tuple[int, int]:
    layers = {arch.layer_of(k) for k in weights if k.startswith(arch.layer_prefix)}
    k = max(layers) + 1 if layers else 0
    m = int(weights[f"{arch.layer_prefix}0.intermediate.dense.weight"].shape[0]) if k else arch.ffn
    return k, m


def count_params(weights: dict[str, np.ndarray], arch: Arch) -> dict:
    groups = {"embeddings": 0, "encoder": 0, "head": 0, "other": 0}
    total = 0
    for k, v in weights.items():
        if arch.is_buffer(k):
            continue
        n = int(np.prod(v.shape))
        total += n
        if arch.is_frozen(k):
            groups["embeddings"] += n
        elif k.startswith(arch.layer_prefix):
            groups["encoder"] += n
        elif k.startswith("classifier."):
            groups["head"] += n
        else:
            groups["other"] += n
    k, m = structure_from_weights(weights, arch)
    emb = groups["embeddings"]
    return {"total": total, **groups, "non_embedding": total - emb, "num_layers": k, "intermediate_size": m,
            "fraction_of_full": total / arch.full_params, "reduction_vs_full": 1 - total / arch.full_params,
            "non_embedding_reduction_vs_full": 1 - (total - emb) / (arch.full_params - arch.embedding_params)}

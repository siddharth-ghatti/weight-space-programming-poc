"""Structural pruning spec: which encoder layers and which feed-forward units survive.

The editor saves tensor edits at full shape; the harness applies a Structure when it builds the
edited model for a forward pass and when it exports the final checkpoint. A pruned checkpoint is a
stock `BertForSequenceClassification` with `num_hidden_layers = k` and `intermediate_size = m`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FULL_LAYERS = 12
FULL_FFN = 3072
HIDDEN = 768

# Parameter accounting for bert-base-uncased + an 8-way head (verified against the checkpoint).
EMBEDDING_PARAMS = 23_837_184
POOLER_PARAMS = 590_592
CLASSIFIER_PARAMS = 6_152
ATTENTION_PARAMS = 2_366_208   # q/k/v/o weights+biases, both LayerNorms, output.dense.bias
FFN_UNIT_PARAMS = 1_537        # 768 (in) + 1 (bias) + 768 (out) per intermediate unit
FIXED_PARAMS = EMBEDDING_PARAMS + POOLER_PARAMS + CLASSIFIER_PARAMS  # 24,433,928
FULL_PARAMS = FIXED_PARAMS + FULL_LAYERS * (ATTENTION_PARAMS + FFN_UNIT_PARAMS * FULL_FFN)  # 109,488,392

LAYER_PREFIX = "bert.encoder.layer."


class StructureError(ValueError):
    pass


@dataclass(frozen=True)
class Structure:
    layers: tuple[int, ...]                  # original layer ids, strictly increasing
    ffn_keep: dict[int, tuple[int, ...]]     # original layer id -> kept intermediate unit ids (sorted), same length everywhere

    @property
    def m(self) -> int:
        return len(self.ffn_keep[self.layers[0]])

    @property
    def k(self) -> int:
        return len(self.layers)

    def is_full(self) -> bool:
        full = tuple(range(FULL_FFN))
        return self.layers == tuple(range(FULL_LAYERS)) and all(self.ffn_keep[l] == full for l in self.layers)

    def to_dict(self) -> dict:
        return {"version": 1, "layers": list(self.layers), "m": self.m,
                "ffn_keep": {str(l): list(self.ffn_keep[l]) for l in self.layers}}

    def layer_map(self) -> dict[int, int]:
        """new layer index (1-based hidden-state index) -> original layer id."""
        return {i + 1: l for i, l in enumerate(self.layers)}


def default_structure() -> Structure:
    return Structure(tuple(range(FULL_LAYERS)), {l: tuple(range(FULL_FFN)) for l in range(FULL_LAYERS)})


def validate_structure(d: dict) -> Structure:
    if not isinstance(d, dict):
        raise StructureError("structure must be a JSON object")
    layers = d.get("layers")
    if not isinstance(layers, list) or not layers:
        raise StructureError("structure.layers must be a non-empty list of original layer ids")
    try:
        layers = [int(x) for x in layers]
    except (TypeError, ValueError):
        raise StructureError("structure.layers must contain integers")
    if any(l < 0 or l >= FULL_LAYERS for l in layers):
        raise StructureError(f"structure.layers must be within 0..{FULL_LAYERS - 1}")
    if any(b <= a for a, b in zip(layers, layers[1:])):
        raise StructureError("structure.layers must be strictly increasing (drop layers, do not reorder or repeat)")
    raw = d.get("ffn_keep")
    if raw is None:
        raw = {str(l): list(range(FULL_FFN)) for l in layers}
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
        if any(x < 0 or x >= FULL_FFN for x in u):
            raise StructureError(f"ffn_keep[{l}] unit ids must be within 0..{FULL_FFN - 1}")
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
        raise StructureError(f"all kept layers must keep the same number of FFN units (stock BERT has one intermediate_size); got sizes {sorted(sizes)}")
    s = Structure(tuple(layers), keep)
    if "m" in d and d["m"] is not None and int(d["m"]) != s.m:
        raise StructureError(f"structure.m={d['m']} does not match the ffn_keep lists (length {s.m})")
    return s


def load_structure(work_dir: Path) -> Structure:
    p = Path(work_dir) / "structure.json"
    if not p.exists():
        return default_structure()
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise StructureError(f"structure.json is not valid JSON: {e}")
    return validate_structure(d)


def structure_hash(s: Structure) -> str:
    if s.is_full():
        return "full"
    return hashlib.sha256(json.dumps(s.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def param_count(s: Structure) -> dict:
    per_layer = ATTENTION_PARAMS + FFN_UNIT_PARAMS * s.m
    encoder = per_layer * s.k
    total = FIXED_PARAMS + encoder
    return {
        "total": total,
        "embeddings": EMBEDDING_PARAMS,
        "encoder": encoder,
        "per_layer": per_layer,
        "pooler": POOLER_PARAMS,
        "classifier": CLASSIFIER_PARAMS,
        "non_embedding": total - EMBEDDING_PARAMS,
        "num_layers": s.k,
        "intermediate_size": s.m,
        "fraction_of_full": total / FULL_PARAMS,
        "reduction_vs_full": 1 - total / FULL_PARAMS,
    }


def is_full_shaped(weights: dict[str, np.ndarray]) -> bool:
    last = f"{LAYER_PREFIX}{FULL_LAYERS - 1}.attention.self.query.weight"
    if last not in weights:
        return False
    for l in range(FULL_LAYERS):
        w = weights.get(f"{LAYER_PREFIX}{l}.intermediate.dense.weight")
        if w is None or w.shape[0] != FULL_FFN:
            return False
    return True


def prune_weights(weights: dict[str, np.ndarray], s: Structure) -> dict[str, np.ndarray]:
    """Keep s.layers (renumbered 0..k-1) and, per kept layer, the FFN units in s.ffn_keep."""
    if not is_full_shaped(weights):
        raise StructureError("prune_weights expects full-shaped (12-layer, 3072-unit) weights; refusing to prune twice")
    out: dict[str, np.ndarray] = {k: v for k, v in weights.items() if not k.startswith(LAYER_PREFIX)}
    for new_i, old_i in enumerate(s.layers):
        old_p, new_p = f"{LAYER_PREFIX}{old_i}.", f"{LAYER_PREFIX}{new_i}."
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


def structure_from_weights(weights: dict[str, np.ndarray]) -> tuple[int, int]:
    """(num_layers, intermediate_size) of an already-pruned or full weight dict."""
    layers = {int(k.split(".")[3]) for k in weights if k.startswith(LAYER_PREFIX)}
    k = max(layers) + 1 if layers else 0
    m = int(weights[f"{LAYER_PREFIX}0.intermediate.dense.weight"].shape[0]) if k else FULL_FFN
    return k, m


def count_params(weights: dict[str, np.ndarray]) -> dict:
    groups = {"embeddings": 0, "encoder": 0, "pooler": 0, "classifier": 0}
    total = 0
    for k, v in weights.items():
        if k.startswith("cls."):
            continue
        n = int(np.prod(v.shape))
        total += n
        if k.startswith("bert.embeddings."):
            groups["embeddings"] += n
        elif k.startswith(LAYER_PREFIX):
            groups["encoder"] += n
        elif k.startswith("bert.pooler."):
            groups["pooler"] += n
        elif k.startswith("classifier."):
            groups["classifier"] += n
    k, m = structure_from_weights(weights)
    return {"total": total, **groups, "non_embedding": total - groups["embeddings"],
            "num_layers": k, "intermediate_size": m, "fraction_of_full": total / FULL_PARAMS,
            "reduction_vs_full": 1 - total / FULL_PARAMS}

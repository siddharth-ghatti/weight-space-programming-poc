"""Sandbox prelude for the reinsurance pruning harness: tensors of the START checkpoint at full shape,
edits at full shape, structure helpers. No helper performs any forward computation."""

from __future__ import annotations

import json
from pathlib import Path

from wspr.arch import Arch

PRELUDE_TEMPLATE = r'''
# ---- harness prelude (do not rely on anything not documented in the task) ----
import json as _json, os as _os, sys as _sys
import numpy as np
from pathlib import Path as _Path
from safetensors import safe_open as _safe_open

WORK = _Path(__WORK__)
START_SAFETENSORS = __BASE__
TOKENIZER_JSON = __TOKJSON__
LABELS = __LABELS__
NUM_LABELS = len(LABELS)
HIDDEN = __HIDDEN__
NUM_LAYERS = __NUM_LAYERS__
FFN = __FFN__
NUM_HEADS = __HEADS__
MAX_LEN = __MAX_LEN__
FULL_PARAMS = __FULL_PARAMS__
FIXED_PARAMS = __FIXED_PARAMS__
EMBEDDING_PARAMS = __EMBEDDING_PARAMS__
LAYER_ATTENTION_PARAMS = __ATTN_PARAMS__
FFN_UNIT_PARAMS = __FFN_UNIT_PARAMS__
_EDITS = WORK / "edits"
_EDITS.mkdir(exist_ok=True)
_STRUCT = WORK / "structure.json"
_FROZEN_PREFIX = __FROZEN_PREFIX__
_BUFFERS = set(__BUFFERS__)

with _safe_open(START_SAFETENSORS, "np") as _f:
    _SHAPES = {k: tuple(_f.get_slice(k).get_shape()) for k in _f.keys() if k not in _BUFFERS}

def catalog():
    """List of dicts: name, shape, editable."""
    return [{"name": n, "shape": list(_SHAPES[n]), "editable": not n.startswith(_FROZEN_PREFIX)} for n in sorted(_SHAPES)]

def base(name):
    """Start-model tensor (the model you are shrinking), float32 copy at full shape."""
    if name in _SHAPES:
        with _safe_open(START_SAFETENSORS, "np") as f:
            return f.get_tensor(name).astype(np.float32)
    raise KeyError(f"unknown tensor {name!r}; see catalog()")

def W(name):
    """Current tensor at full shape: your saved edit if one exists, else the start tensor."""
    p = _EDITS / f"{name}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    return base(name)

def save_edit(name, arr):
    """Validate and save an edited tensor at its FULL original shape. Refuses frozen tensors, shape
    mismatch, non-finite values. Pruning is applied by the harness from structure.json."""
    if name not in _SHAPES:
        raise ValueError(f"unknown tensor {name!r}; see catalog()")
    if name.startswith(_FROZEN_PREFIX):
        raise ValueError(f"{name!r} is frozen (embeddings cannot be edited)")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != _SHAPES[name]:
        raise ValueError(f"shape mismatch for {name!r}: got {arr.shape}, expected {_SHAPES[name]} (edits are saved at full shape; the harness slices kept FFN units)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name!r} contains non-finite values")
    np.save(_EDITS / f"{name}.npy", np.ascontiguousarray(arr))
    return f"saved edit for {name} {arr.shape}"

def drop_edit(name):
    p = _EDITS / f"{name}.npy"
    if p.exists():
        p.unlink(); return f"dropped edit for {name}"
    return f"no edit saved for {name}"

def edits():
    """Names of tensors that currently have a saved edit."""
    return sorted(p.stem for p in _EDITS.glob("*.npy"))

def practice():
    """Practice paragraphs with labels: list of {id, text, label}."""
    return [_json.loads(l) for l in (WORK / "practice.jsonl").read_text().splitlines() if l.strip()]

_tok = None
def _tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(TOKENIZER_JSON)
    return _tok

def tokenize(texts, max_length=MAX_LEN):
    """Byte-level BPE tokenization as the model sees it (with <s> ... </s>). Returns list of {ids, tokens}. Not a forward pass."""
    tok = _tokenizer()
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for enc in tok.encode_batch(texts):
        out.append({"ids": enc.ids[:max_length], "tokens": enc.tokens[:max_length]})
    return out

def vocab():
    """Dict token -> id."""
    return _tokenizer().get_vocab()

def pass_dirs():
    """Folders of completed forward passes (pass_1, pass_2, ...)."""
    return sorted(str(p) for p in WORK.glob("pass_*"))

# ---- structure (which layers / FFN units survive) ----

def _validate_structure(d):
    layers = d.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("layers must be a non-empty list of original layer ids")
    layers = [int(x) for x in layers]
    if any(l < 0 or l >= NUM_LAYERS for l in layers):
        raise ValueError(f"layers must be within 0..{NUM_LAYERS - 1}")
    if any(b <= a for a, b in zip(layers, layers[1:])):
        raise ValueError("layers must be strictly increasing (drop layers; do not reorder or repeat)")
    raw = d.get("ffn_keep") or {str(l): list(range(FFN)) for l in layers}
    keep = {}
    for key, units in raw.items():
        l = int(key)
        if l not in layers:
            raise ValueError(f"ffn_keep has an entry for layer {l}, which is not kept")
        u = [int(x) for x in units]
        if not u or any(x < 0 or x >= FFN for x in u):
            raise ValueError(f"ffn_keep[{l}] must be non-empty unit ids within 0..{FFN - 1}")
        if len(set(u)) != len(u) or u != sorted(u):
            raise ValueError(f"ffn_keep[{l}] must be sorted and unique")
        keep[l] = u
    missing = [l for l in layers if l not in keep]
    if missing:
        raise ValueError(f"ffn_keep is missing entries for kept layers {missing}")
    if len({len(keep[l]) for l in layers}) != 1:
        raise ValueError("all kept layers must keep the same number of FFN units (the checkpoint has a single intermediate_size)")
    m = len(keep[layers[0]])
    return {"version": 1, "layers": layers, "m": m, "ffn_keep": {str(l): keep[l] for l in layers}}

def _full():
    return {"version": 1, "layers": list(range(NUM_LAYERS)), "m": FFN, "ffn_keep": {str(l): list(range(FFN)) for l in range(NUM_LAYERS)}}

def structure():
    """Current structure: {layers, m, ffn_keep}. Default = all layers, all FFN units."""
    if _STRUCT.exists():
        return _validate_structure(_json.loads(_STRUCT.read_text()))
    return _full()

def param_count(struct=None):
    """Exported parameter count for a structure (default: the current one). Pure arithmetic."""
    s = struct if struct is not None else structure()
    layers = [int(x) for x in s["layers"]]
    m = int(s.get("m") or len(next(iter(s["ffn_keep"].values()))))
    per_layer = LAYER_ATTENTION_PARAMS + FFN_UNIT_PARAMS * m
    total = FIXED_PARAMS + per_layer * len(layers)
    ne_full = FULL_PARAMS - EMBEDDING_PARAMS
    return {"total": total, "non_embedding": total - EMBEDDING_PARAMS, "num_layers": len(layers), "intermediate_size": m,
            "per_layer": per_layer, "fraction_of_full": total / FULL_PARAMS, "reduction_vs_full": 1 - total / FULL_PARAMS,
            "non_embedding_reduction_vs_full": 1 - (total - EMBEDDING_PARAMS) / ne_full}

def set_structure(layers=None, ffn_keep=None):
    """Write structure.json. layers: original layer ids to keep (default: current). ffn_keep: None
    (current units for kept layers, all units for newly kept ones), an int m (units 0..m-1 for every
    kept layer), a list of unit ids (same for every kept layer), or {layer: [unit ids]}."""
    try:
        cur = structure()
    except ValueError:  # an invalid structure.json must not block repairing it
        cur = _full()
    layers = cur["layers"] if layers is None else [int(x) for x in layers]
    if ffn_keep is None:
        keep = {str(l): cur["ffn_keep"].get(str(l), list(range(FFN))) for l in layers}
    elif isinstance(ffn_keep, (int, np.integer)):
        keep = {str(l): list(range(int(ffn_keep))) for l in layers}
    elif isinstance(ffn_keep, dict):
        keep = {str(int(k)): [int(x) for x in v] for k, v in ffn_keep.items()}
    else:
        units = sorted(int(x) for x in ffn_keep)
        keep = {str(l): units for l in layers}
    d = _validate_structure({"layers": layers, "ffn_keep": keep})
    _STRUCT.write_text(_json.dumps(d))
    pc = param_count(d)
    return (f"structure: layers={d['layers']} m={d['m']} params={pc['total']:,} "
            f"({100 * pc['fraction_of_full']:.1f}% of {FULL_PARAMS:,}; {100 * pc['reduction_vs_full']:.1f}% smaller; "
            f"non-embedding {100 * pc['non_embedding_reduction_vs_full']:.1f}% smaller)")

def clear_structure():
    """Back to the full structure (all layers, all FFN units)."""
    if _STRUCT.exists():
        _STRUCT.unlink()
    return "structure cleared (full model)"

def kept_units(layer):
    """Unit ids kept in the FFN of an original layer under the current structure."""
    return list(structure()["ffn_keep"].get(str(int(layer)), []))

np.set_printoptions(precision=4, suppress=True, linewidth=140)
# ---- end prelude ----
'''


def build_prelude(work_dir: Path, arch: Arch, max_len: int = 512) -> str:
    return (
        PRELUDE_TEMPLATE
        .replace("__WORK__", json.dumps(str(work_dir)))
        .replace("__BASE__", json.dumps(str(arch.model_dir / "model.safetensors")))
        .replace("__TOKJSON__", json.dumps(str(arch.model_dir / "tokenizer.json")))
        .replace("__LABELS__", json.dumps(list(arch.labels)))
        .replace("__HIDDEN__", str(arch.hidden))
        .replace("__NUM_LAYERS__", str(arch.num_layers))
        .replace("__FFN__", str(arch.ffn))
        .replace("__HEADS__", str(arch.heads))
        .replace("__MAX_LEN__", str(max_len))
        .replace("__FULL_PARAMS__", str(arch.full_params))
        .replace("__FIXED_PARAMS__", str(arch.fixed_params))
        .replace("__EMBEDDING_PARAMS__", str(arch.embedding_params))
        .replace("__ATTN_PARAMS__", str(arch.attention_params))
        .replace("__FFN_UNIT_PARAMS__", str(arch.ffn_unit_params))
        .replace("__FROZEN_PREFIX__", json.dumps(arch.frozen_prefix))
        .replace("__BUFFERS__", json.dumps(list(arch.buffer_keys)))
    )


def prelude_docs(arch: Arch) -> str:
    return f"""\
Every code run starts with a prelude that defines (numpy is imported as `np`):

- `catalog()` -> list of {{name, shape, editable}} for every tensor of the start model.
- `base(name)` -> float32 copy of the START model's tensor (the model you are shrinking), full shape.
- `W(name)` -> the current tensor at full shape: your saved edit if one exists, else the start tensor.
- `save_edit(name, array)` -> validates (editable, exact FULL shape, finite) and saves to `edits/<name>.npy`.
  Saving the same name again overwrites. Edits are always at the full original shape; the harness
  slices kept FFN units and renumbers kept layers when it builds the edited model and the export.
- `drop_edit(name)`, `edits()` -> manage/list saved edits.
- `practice()` -> list of {{id, text, label}} practice paragraphs (also at `practice.jsonl`).
- `tokenize(texts)` -> [{{ids, tokens}}] using the model's byte-level BPE tokenizer (`<s>` ... `</s>`).
- `vocab()` -> token -> id dict.
- `pass_dirs()` -> folders of completed forward passes; see each folder's files.
- `WORK` -> Path of the persistent working folder (cwd). Write notes, files, arrays freely.
- `LABELS`, `NUM_LABELS`, `HIDDEN` ({arch.hidden}), `NUM_LAYERS` ({arch.num_layers}), `FFN` ({arch.ffn}),
  `NUM_HEADS` ({arch.heads}), `MAX_LEN` (512) constants.

Each run is a fresh process; only files in the working folder persist between runs.
Available packages: numpy, scipy, safetensors, tokenizers, and the standard library. No torch, no
transformers, no network.
"""


def structure_docs(arch: Arch) -> str:
    return f"""\
Structure helpers (the pruning spec lives in `structure.json` in the working folder):

- `structure()` -> {{layers, m, ffn_keep}}: original layer ids kept (in order) and, per kept layer,
  the sorted ids of the FFN units kept. Default: all {arch.num_layers} layers, all {arch.ffn:,} units.
- `set_structure(layers=None, ffn_keep=None)` -> writes `structure.json` and returns the resulting
  parameter count. `layers`: original ids to keep, strictly increasing. `ffn_keep`: an int m (units
  0..m-1 everywhere), a list of unit ids (same everywhere), or {{layer: [unit ids]}}; every kept layer
  must keep the same number m of units. The edited model used by forward passes and by the final
  export is: start + your saved edits, then kept layers renumbered 0..k-1 and FFN units sliced.
- `clear_structure()`, `kept_units(layer)`, `param_count(struct=None)` -> exported parameter count.

Parameter accounting of the exported checkpoint (stock `RobertaForSequenceClassification`):
  total = {arch.fixed_params:,} + k * ({arch.attention_params:,} + {arch.ffn_unit_params:,} * m)
  where k = layers kept, m = FFN units kept per layer; the fixed part is the embeddings
  ({arch.embedding_params:,}: {arch.vocab:,} word vectors, {arch.max_pos} positions, LayerNorm; frozen and never
  pruned) and the classification head ({arch.head_params:,}). The full model is {arch.full_params:,}, of
  which {arch.full_params - arch.embedding_params:,} are outside the embeddings.
Slicing FFN unit i removes row i of `intermediate.dense.weight`/`bias` and column i of
`output.dense.weight`; `output.dense.bias` and both LayerNorms of a kept layer stay at full size.
Attention heads cannot be removed from a stock checkpoint: zeroing a head saves nothing.
"""

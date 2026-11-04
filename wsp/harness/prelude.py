"""Code prepended to every sandbox run. Gives the editor read access to the weights, the tokenizer
and the practice set, and a `save_edit` that writes validated edits to /work/edits.

No helper here performs any forward computation.
"""

from __future__ import annotations

import json
from pathlib import Path

from wsp.labels import LABELS
from wsp.model.weights import BASE_DIR, HIDDEN, NUM_LABELS, ROOT


PRELUDE_TEMPLATE = r'''
# ---- harness prelude (do not rely on anything not documented in the task) ----
import json as _json, os as _os, sys as _sys
import numpy as np
from pathlib import Path as _Path
from safetensors import safe_open as _safe_open

WORK = _Path(__WORK__)
BASE_SAFETENSORS = __BASE__
TOKENIZER_JSON = __TOKJSON__
LABELS = __LABELS__
NUM_LABELS = len(LABELS)
HIDDEN = __HIDDEN__
_EDITS = WORK / "edits"
_EDITS.mkdir(exist_ok=True)
_LEGACY = {"LayerNorm.weight": "LayerNorm.gamma", "LayerNorm.bias": "LayerNorm.beta"}
_ADDED = {"classifier.weight": (NUM_LABELS, HIDDEN), "classifier.bias": (NUM_LABELS,)}
_FROZEN_PREFIX = "bert.embeddings."
_EXTRA_PREFIX = "cls."

def _raw_key(name):
    for new, old in _LEGACY.items():
        if name.endswith(new):
            return name[:-len(new)] + old
    return name

def _norm_key(raw):
    for new, old in _LEGACY.items():
        if raw.endswith(old):
            return raw[:-len(old)] + new
    return raw

with _safe_open(BASE_SAFETENSORS, "np") as _f:
    _SHAPES = {_norm_key(k): tuple(_f.get_slice(k).get_shape()) for k in _f.keys()}
_SHAPES.update(_ADDED)

def catalog():
    """List of dicts: name, shape, editable, in_model."""
    out = []
    for n in sorted(_SHAPES):
        editable = not n.startswith(_FROZEN_PREFIX) and not n.startswith(_EXTRA_PREFIX)
        out.append({"name": n, "shape": list(_SHAPES[n]), "editable": editable, "in_model": not n.startswith(_EXTRA_PREFIX)})
    return out

def base(name):
    """Base (unedited) tensor as a float32 numpy array (a copy; edit freely)."""
    if name in _ADDED:
        return np.zeros(_ADDED[name], dtype=np.float32)
    if name not in _SHAPES:
        raise KeyError(f"unknown tensor {name!r}; see catalog()")
    with _safe_open(BASE_SAFETENSORS, "np") as f:
        return f.get_tensor(_raw_key(name)).astype(np.float32)

def W(name):
    """Current tensor: the saved edit if one exists, else the base tensor."""
    p = _EDITS / f"{name}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    return base(name)

def save_edit(name, arr):
    """Validate and save an edited tensor. Refuses frozen/extra tensors, shape mismatch, non-finite."""
    if name not in _SHAPES:
        raise ValueError(f"unknown tensor {name!r}; see catalog()")
    if name.startswith(_FROZEN_PREFIX):
        raise ValueError(f"{name!r} is frozen (embeddings cannot be edited)")
    if name.startswith(_EXTRA_PREFIX):
        raise ValueError(f"{name!r} is not part of the classification model and cannot be edited")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != _SHAPES[name]:
        raise ValueError(f"shape mismatch for {name!r}: got {arr.shape}, expected {_SHAPES[name]}")
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
    """Practice sentences with labels: list of {id, text, label}."""
    return [_json.loads(l) for l in (WORK / "practice.jsonl").read_text().splitlines() if l.strip()]

_tok = None
def _tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(TOKENIZER_JSON)
    return _tok

def tokenize(texts, max_length=128):
    """WordPiece tokenization as the model sees it (uncased, [CLS] ... [SEP]). Returns list of
    {ids, tokens}. This reads the tokenizer only; it is not a forward pass."""
    tok = _tokenizer()
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for enc in tok.encode_batch(texts):
        ids = enc.ids[:max_length]
        out.append({"ids": ids, "tokens": enc.tokens[:max_length]})
    return out

def vocab():
    """Dict token -> id (30522 entries). Row i of bert.embeddings.word_embeddings.weight is token i."""
    return _tokenizer().get_vocab()

def pass_dirs():
    """Folders of completed forward passes (pass_1, pass_2, ...)."""
    return sorted(str(p) for p in WORK.glob("pass_*"))

np.set_printoptions(precision=4, suppress=True, linewidth=140)
# ---- end prelude ----
'''


def build_prelude(work_dir: Path) -> str:
    return (
        PRELUDE_TEMPLATE
        .replace("__WORK__", json.dumps(str(work_dir)))
        .replace("__BASE__", json.dumps(str(BASE_DIR / "model.safetensors")))
        .replace("__TOKJSON__", json.dumps(str(BASE_DIR / "tokenizer.json")))
        .replace("__LABELS__", json.dumps(LABELS))
        .replace("__HIDDEN__", str(HIDDEN))
    )


# Documentation of the prelude API, inserted into the task prompt.
PRELUDE_DOCS = """\
Every code run starts with a prelude that defines (numpy is imported as `np`):

- `catalog()` -> list of {name, shape, editable, in_model} for every tensor.
- `base(name)` -> float32 numpy copy of the base (unedited) tensor.
- `W(name)` -> the current tensor: your saved edit if one exists, else the base tensor.
- `save_edit(name, array)` -> validates (editable, exact shape, finite) and saves to `edits/<name>.npy`.
  Saving the same name again overwrites. The *edited model* used by forward passes and by the final
  evaluation is base + all saved edits.
- `drop_edit(name)`, `edits()` -> manage/list saved edits.
- `practice()` -> list of {id, text, label} practice sentences (also at `practice.jsonl`).
- `tokenize(texts)` -> [{ids, tokens}] using the model's WordPiece tokenizer (uncased).
- `vocab()` -> token -> id dict. Word-embedding row i is token id i.
- `pass_dirs()` -> folders of completed forward passes; see each folder's files.
- `WORK` -> Path of the persistent working folder (cwd). Write notes, files, arrays freely.
- `LABELS`, `NUM_LABELS`, `HIDDEN` constants.

Each run is a fresh process; only files in the working folder persist between runs.
Available packages: numpy, scipy, safetensors, tokenizers, and the standard library. No torch, no
transformers, no network.
"""

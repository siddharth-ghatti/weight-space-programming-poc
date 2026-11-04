"""The starting model: bert-base-uncased + a verified edit bundle from a `wsp` run (e.g. runs/v1/p50)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from wsp.model import weights as W

ROOT = W.ROOT


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint_dir(model_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with safe_open(str(Path(model_dir) / "model.safetensors"), "np") as f:
        for k in f.keys():
            out[W.normalize_name(k)] = np.ascontiguousarray(f.get_tensor(k).astype(np.float32))
    return out


def weights_hash(weights: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(weights):
        if W.is_extra(k):
            continue
        h.update(k.encode()); h.update(W.sha256(weights[k]).encode())
    return h.hexdigest()


def load_start(init_from: str | None) -> tuple[dict[str, np.ndarray], dict]:
    """Weights to start from (classifier model tensors only, no cls.*) and a provenance record."""
    base = W.load_base()
    if not init_from:
        start = {k: v for k, v in base.items() if not W.is_extra(k)}
        return start, {"base_model": "bert-base-uncased", "init_from": None, "start_model_sha256": weights_hash(start)}
    run = ROOT / init_from
    edit_dir = run / "edit"
    if not (edit_dir / "edit.npz").exists():
        raise FileNotFoundError(f"{edit_dir} has no edit.npz; init_from must point at a finished wsp run")
    edits = W.load_edit_bundle(edit_dir, base, verify=True)
    start = {k: v for k, v in W.apply_edits(base, edits).items() if not W.is_extra(k)}
    ck = run / "model" / "model.safetensors"
    if ck.exists():  # cross-check against the exported checkpoint of that run
        exported = load_checkpoint_dir(ck.parent)
        bad = [k for k in start if k not in exported or W.sha256(exported[k]) != W.sha256(start[k])]
        if bad:
            raise RuntimeError(f"start weights differ from {ck} for {bad[:5]}")
    prov = {
        "base_model": "bert-base-uncased",
        "init_from": init_from,
        "init_edit_manifest_sha256": file_sha256(edit_dir / "manifest.json"),
        "init_edit_npz_sha256": file_sha256(edit_dir / "edit.npz"),
        "start_model_sha256": weights_hash(start),
    }
    scores = run / "eval" / "scores.json"
    if scores.exists():
        s = json.loads(scores.read_text())
        prov["init_test_accuracy"] = s["test"]["accuracy"]
        prov["init_test_macro_f1"] = s["test"]["macro_f1"]
    return start, prov

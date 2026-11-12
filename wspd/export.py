"""Edit bundle (the exact diff), baked checkpoint export, rebuild check, model card."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from wspd.common import ANSWER_WORDS, MODEL_DIR
from wspd.reader import down_proj


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save_edit_bundle(edit_dir: Path, layer: int, cols: list[int], D: torch.Tensor, provenance: dict, extra: dict) -> dict:
    """edit.npz (D fp32 and the column ids; ignored by git) + manifest.json (hashes, provenance; committed)."""
    edit_dir.mkdir(parents=True, exist_ok=True)
    Dn = D.detach().float().cpu().numpy()
    np.savez(edit_dir / "edit.npz", D=Dn, cols=np.asarray(cols, dtype=np.int64))
    man = {"method": "direct closed-form column edit of one mlp.down_proj (no gradient descent)",
           "base_model": provenance.get("hub_repo"), "base_commit": provenance.get("hub_commit"),
           "base_safetensors_sha256": provenance.get("safetensors_sha256"),
           "layer": layer, "n_columns": len(cols), "columns": cols, "D_shape": list(Dn.shape), "D_sha256": sha256_arr(Dn),
           "D_fro_norm": float(np.linalg.norm(Dn)), "weights_in_edit": int(Dn.size), **extra}
    (edit_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def export_checkpoint(student, out_dir: Path, manifest: dict, card: str) -> Path:
    """Save the baked model as a stock checkpoint (fp16 weights), tokenizer files, manifest and card."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    student.model.save_pretrained(str(out_dir), safe_serialization=True)
    student.tok.save_pretrained(str(out_dir))
    for fn in ("chat_template.jinja", "LICENSE"):
        src = student.model_dir / fn
        if src.exists() and not (out_dir / fn).exists():
            shutil.copy(src, out_dir / fn)
    (out_dir / "edit_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "README.md").write_text(card)
    return out_dir


@torch.no_grad()
def rebuild_check(baked_model_dir: Path, edit_dir: Path, base_dir: Path = MODEL_DIR) -> dict:
    """Base weights + edit.npz, rounded to the checkpoint dtype, must equal the baked checkpoint's edited matrix
    exactly; every other tensor must be identical. Loads on CPU."""
    from transformers import AutoModelForCausalLM
    man = json.loads((edit_dir / "manifest.json").read_text())
    z = np.load(edit_dir / "edit.npz")
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float16)
    baked = AutoModelForCausalLM.from_pretrained(str(baked_model_dir), dtype=torch.float16)
    W = down_proj(base, man["layer"]).weight
    cols = torch.as_tensor(z["cols"])
    W[:, cols] = (W[:, cols].float() + torch.as_tensor(z["D"])).half()
    sb, sk = base.state_dict(), baked.state_dict()
    diff = [k for k in sb if k in sk and not torch.equal(sb[k], sk[k])]
    missing = sorted(set(sb) ^ set(sk))
    return {"identical": not diff and not missing, "tensors_differing": diff[:10], "keys_mismatch": missing[:10]}


def model_card(manifest: dict, scores: dict | None, run_name: str) -> str:
    repo = manifest.get("base_model", "Qwen/Qwen3.5-0.8B"); commit = manifest.get("base_commit") or ""
    s = scores or {}
    gen = s.get("generation", {}); sc = s.get("scored", {})
    lines = [
        "---", f"base_model: {repo}", "language: en", "license: apache-2.0", "pipeline_tag: text-generation",
        "tags:", "- weight-space-programming", "- weight-editing", "- distillation", "- insurance", "- qwen3.5", "---", "",
        "# Qwen3.5-0.8B taught an insurance clause classification by a closed-form weight edit (no gradient descent)", "",
        f"[{repo}](https://huggingface.co/{repo})" + (f" (commit `{commit[:12]}`)" if commit else "") +
        f" with {manifest.get('n_columns')} columns of `layers.{manifest.get('layer')}.mlp.down_proj` rewritten "
        f"({manifest.get('weights_in_edit', 0):,} weights, {100 * manifest.get('weights_in_edit', 0) / max(1, manifest.get('model_params', 1)):.3f}% of the model). "
        "A Jev-like teacher (calibrated decision heads on the frozen reader, fitted on 1,496 labelled insurance sentences) "
        "supplied target probabilities; one ridge least-squares solve per round wrote them into the weights, with a dose "
        "chosen on kept-back rows and a cap on the disturbance of ordinary text. No training loop, no optimizer, no LoRA.", "",
        f"Prompted with a sentence, the question and the eight options, the model answers with one of: "
        f"{', '.join(f'`{w}`' for w in ANSWER_WORDS)}.", "",
        "| | value |", "|---|---|",
        f"| held-out test accuracy, generated first word (1,779 sentences) | {100 * gen.get('acc', 0):.2f}% |",
        f"| held-out macro-F1 | {sc.get('macro_f1', 0):.4f} |",
        f"| agreement with the teacher | {100 * gen.get('agree', 0):.1f}% |",
        f"| weights changed | {manifest.get('weights_changed', 0):,} |",
        f"| run | {run_name} |", "",
        "`edit_manifest.json` records the exact diff against the base checkpoint (layer, columns, hash of the column "
        "update). The run folder in the repository has the full log, the teacher, the LoRA comparison and the forgetting suite.", "",
        "## Caveats", "",
        "- Research artifact from a proof of concept: one base model, synthetic insurance data (generated for the study), "
        "a single edited matrix. Not production-ready.",
        "- The model is a general chat LM with one skill installed; outside the prompt format above it behaves like the base model "
        "(the run's forgetting suite reports how closely).",
        "- Probabilities over the answer words are inherited from the teacher's calibration; re-calibrate before using them as confidences.", "",
    ]
    return "\n".join(lines)

"""Download the reader (Qwen3.5-0.8B), pinned to a commit sha, into models/qwen3.5-0.8b/.

    uv run python scripts/distill/fetch_model.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
REPO = "Qwen/Qwen3.5-0.8B"
SHA = "2fc06364715b967f1860aea9cf38778875588b17"
LOCAL = "qwen3.5-0.8b"
ALLOW = ["config.json", "*.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json", "LICENSE", "README.md"]


def main() -> None:
    api = HfApi()
    out = ROOT / "models" / LOCAL
    info = api.model_info(REPO, revision=SHA)
    snapshot_download(REPO, revision=info.sha, local_dir=str(out), allow_patterns=ALLOW)
    sts = sorted(out.glob("*.safetensors"))
    prov = {"hub_repo": REPO, "hub_commit": info.sha,
            "license": getattr(info.card_data, "license", None) if info.card_data else None,
            "safetensors_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sts},
            "files": sorted(p.name for p in out.iterdir() if p.is_file())}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(LOCAL, "<-", REPO, "@", info.sha, f"{sum(p.stat().st_size for p in sts) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

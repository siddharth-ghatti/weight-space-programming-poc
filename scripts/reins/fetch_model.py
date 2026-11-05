"""Download the start model (ClimateBERT TCFD classifier) and its MLM base, pinned to commit shas.

    uv run python scripts/reins/fetch_model.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "climatebert-tcfd": ("climatebert/distilroberta-base-climate-tcfd", "970630beedc21db81a84156448ad2e3ac860153d"),
    "climatebert-base-f": ("climatebert/distilroberta-base-climate-f", None),
}
ALLOW = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "special_tokens_map.json", "added_tokens.json", "README.md"]


def main() -> None:
    api = HfApi()
    for local, (repo, sha) in MODELS.items():
        out = ROOT / "models" / local
        info = api.model_info(repo, revision=sha)
        snapshot_download(repo, revision=info.sha, local_dir=str(out), allow_patterns=ALLOW)
        st = out / "model.safetensors"
        prov = {"hub_repo": repo, "hub_commit": info.sha, "license": getattr(info.card_data, "license", None) if info.card_data else None,
                "safetensors_sha256": hashlib.sha256(st.read_bytes()).hexdigest(), "files": sorted(p.name for p in out.iterdir())}
        (out / "provenance.json").write_text(json.dumps(prov, indent=2))
        print(local, "<-", repo, "@", info.sha, f"{st.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

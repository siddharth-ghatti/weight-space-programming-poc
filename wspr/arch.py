"""Architecture description of the start model, read from its checkpoint config.

Everything the harness needs to know about tensor naming, the classification head and the parameter
arithmetic lives here, so the rest of the package is written against `Arch` rather than one model.
Supported family: RoBERTa-style encoders with `RobertaClassificationHead` (dense + out_proj, no pooler).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "climatebert-tcfd"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")


@dataclass(frozen=True)
class Arch:
    model_dir: Path
    model_prefix: str          # "roberta"
    model_type: str            # config.model_type
    hidden: int
    num_layers: int
    ffn: int
    heads: int
    vocab: int
    max_pos: int
    type_vocab: int
    labels: tuple[str, ...]    # in config id order

    # ---- naming
    @property
    def layer_prefix(self) -> str:
        return f"{self.model_prefix}.encoder.layer."

    @property
    def frozen_prefix(self) -> str:
        return f"{self.model_prefix}.embeddings."

    @property
    def buffer_keys(self) -> tuple[str, ...]:
        return (f"{self.model_prefix}.embeddings.position_ids",)

    @property
    def head_keys(self) -> tuple[str, ...]:
        return ("classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias")

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def id2label(self) -> dict[int, str]:
        return {i: l for i, l in enumerate(self.labels)}

    @property
    def label2id(self) -> dict[str, int]:
        return {l: i for i, l in enumerate(self.labels)}

    def is_frozen(self, name: str) -> bool:
        return name.startswith(self.frozen_prefix)

    def is_buffer(self, name: str) -> bool:
        return name in self.buffer_keys

    def layer_of(self, name: str) -> int | None:
        if not name.startswith(self.layer_prefix):
            return None
        return int(name[len(self.layer_prefix):].split(".")[0])

    # ---- parameter arithmetic (verified against the checkpoint in load_arch)
    @property
    def embedding_params(self) -> int:
        return (self.vocab + self.max_pos + self.type_vocab) * self.hidden + 2 * self.hidden

    @property
    def attention_params(self) -> int:
        """q/k/v/o weights+biases, both LayerNorms of a layer, and the FFN output bias."""
        return 4 * (self.hidden * self.hidden + self.hidden) + 2 * (2 * self.hidden) + self.hidden

    @property
    def ffn_unit_params(self) -> int:
        return 2 * self.hidden + 1

    @property
    def head_params(self) -> int:
        return self.hidden * self.hidden + self.hidden + self.hidden * self.num_labels + self.num_labels

    @property
    def fixed_params(self) -> int:
        return self.embedding_params + self.head_params

    @property
    def full_params(self) -> int:
        return self.fixed_params + self.num_layers * (self.attention_params + self.ffn_unit_params * self.ffn)

    def param_count(self, k: int, m: int) -> dict:
        per_layer = self.attention_params + self.ffn_unit_params * m
        total = self.fixed_params + per_layer * k
        return {"total": total, "embeddings": self.embedding_params, "encoder": per_layer * k, "per_layer": per_layer,
                "head": self.head_params, "non_embedding": total - self.embedding_params, "num_layers": k,
                "intermediate_size": m, "fraction_of_full": total / self.full_params,
                "reduction_vs_full": 1 - total / self.full_params,
                "non_embedding_reduction_vs_full": 1 - (total - self.embedding_params) / (self.full_params - self.embedding_params)}

    def describe(self) -> str:
        return (f"{self.model_type} encoder: {self.num_layers} layers, hidden {self.hidden}, {self.heads} heads, "
                f"FFN {self.ffn}, vocab {self.vocab:,}, {self.num_labels} labels; {self.full_params:,} parameters")


def load_arch(model_dir: Path | str = DEFAULT_MODEL_DIR) -> Arch:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    if cfg.get("model_type") != "roberta":
        raise ValueError(f"unsupported model_type {cfg.get('model_type')!r}; wspr supports RoBERTa-style classifiers")
    labels = tuple(cfg["id2label"][str(i)] for i in range(len(cfg["id2label"])))
    return Arch(model_dir=model_dir, model_prefix="roberta", model_type=cfg["model_type"], hidden=int(cfg["hidden_size"]),
                num_layers=int(cfg["num_hidden_layers"]), ffn=int(cfg["intermediate_size"]), heads=int(cfg["num_attention_heads"]),
                vocab=int(cfg["vocab_size"]), max_pos=int(cfg["max_position_embeddings"]), type_vocab=int(cfg.get("type_vocab_size", 1)),
                labels=labels)


def label_block(arch: Arch) -> str:
    return "\n".join(f"{i}. `{l}`" for i, l in enumerate(arch.labels))

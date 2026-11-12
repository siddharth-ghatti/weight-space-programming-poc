"""The reader: Qwen3.5-0.8B as a plain causal LM, frozen. Encoding, next-token log-probabilities, generation,
answer-word ids in both spellings, and the handles the edit needs (text backbone, final norm, lm head)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from wspd.common import ANSWER_WORDS, CAPITALS, LETTERS, MODEL_DIR


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device) -> None:
    d = str(device)
    if d.startswith("cuda"):
        torch.cuda.synchronize()
    elif d.startswith("mps"):
        torch.mps.synchronize()


def text_backbone(model):
    """The module that owns `layers`, `norm` and `embed_tokens` (Qwen3_5TextModel), whichever wrapper loaded."""
    for _, mod in model.named_modules():
        if hasattr(mod, "layers") and hasattr(mod, "norm") and hasattr(mod, "embed_tokens"):
            return mod
    raise RuntimeError("no text backbone with layers/norm/embed_tokens found")


def down_proj(model, layer: int) -> torch.nn.Linear:
    for name, mod in model.named_modules():
        if name.endswith("layers.%d.mlp.down_proj" % layer):
            return mod
    raise KeyError("no layers.%d.mlp.down_proj" % layer)


def n_layers(model) -> int:
    return len([n for n, _ in model.named_modules() if re.search(r"layers\.\d+\.mlp\.down_proj$", n)])


class Student:
    """A plain causal LM answering from a prompt at its next token. Weights frozen; fp16 on GPU/MPS, fp32 on CPU."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: Optional[str] = None, dtype=None, variant: str = "words"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or pick_device())
        self.dtype = dtype or (torch.float32 if self.device.type == "cpu" else torch.float16)
        self.variant = variant
        t = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"     # exact for a causal model: real tokens never see the pads
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir), dtype=self.dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.base = text_backbone(self.model)
        self.head = self.model.get_output_embeddings()
        self.load_seconds = time.perf_counter() - t
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.words = LETTERS if variant == "letters" else ANSWER_WORDS
        self.set_spellings()
        self.city_ids = torch.tensor([self.tok(" " + c, add_special_tokens=False).input_ids[0] for c, _ in CAPITALS],
                                     device=self.device)
        prov = self.model_dir / "provenance.json"
        self.provenance = json.loads(prov.read_text()) if prov.exists() else {}

    # --- answer words
    def set_spellings(self) -> None:
        """Each answer word as a token in two spellings: bare and with a leading space. Only spellings where every
        word is a single token are kept; `choose_spelling` picks the one the reader puts its mass on."""
        self.word_variants: dict[str, torch.Tensor] = {}
        for name, pre in (("bare", ""), ("space", " ")):
            ids = [self.tok(pre + w, add_special_tokens=False).input_ids for w in self.words]
            if all(len(i) == 1 for i in ids):
                self.word_variants[name] = torch.tensor([i[0] for i in ids], device=self.device)
        assert self.word_variants, "no single-token spelling of the answer words"
        self.spelling = "space" if "space" in self.word_variants else next(iter(self.word_variants))
        self.word_ids = self.word_variants[self.spelling]

    @torch.no_grad()
    def choose_spelling(self, enc, n: int = 64) -> dict:
        rows = torch.arange(min(n, len(enc[0])), device=self.device)
        lp = self.last_logp(enc, rows)
        mass = {k: lp[:, ids].exp().sum(-1).mean().item() for k, ids in self.word_variants.items()}
        self.spelling = max(mass, key=mass.get)
        self.word_ids = self.word_variants[self.spelling]
        return {"used": self.spelling, "mass": mass}

    @property
    def answer_prefix(self) -> str:
        return " " if self.spelling == "space" else ""

    # --- encoding and reading
    def encode(self, texts: Sequence[str]):
        enc = self.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
        ids, attn = enc.input_ids.to(self.device), enc.attention_mask.to(self.device)
        return ids, attn, attn.sum(1) - 1

    def hidden(self, ids, attn):
        return self.base(input_ids=ids, attention_mask=attn).last_hidden_state

    def last_logp(self, enc, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full-vocabulary log-probabilities of the next token after each prompt: [n, V]."""
        ids, attn, last = enc
        if rows is not None:
            ids, attn, last = ids[rows], attn[rows], last[rows]
        L = int(last.max().item()) + 1
        h = self.hidden(ids[:, :L], attn[:, :L])
        h = h[torch.arange(len(ids), device=self.device), last]
        return self.head(h).float().log_softmax(-1)

    def text_logp(self, enc) -> torch.Tensor:
        """Next-token log-probabilities at every real position of some ordinary text: [N, V]."""
        ids, attn, _ = enc
        h = self.hidden(ids, attn)[attn.bool()]
        return self.head(h).float().log_softmax(-1)

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], new_tokens: int = 4) -> list[str]:
        out = []
        for p in prompts:   # one at a time, so no padding is ever in play
            ids = self.tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            g = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=new_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            out.append(self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
        return out

    def chat(self, text: str) -> str:
        msgs = [{"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

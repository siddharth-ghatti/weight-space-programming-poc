"""Unit tests for the reinsurance pruning spec: arithmetic vs checkpoint, validation, prelude agreement, equivalence."""
import json
import numpy as np
import pytest
import torch

from wspr.arch import DEFAULT_MODEL_DIR, load_arch
from wspr import structure as S

pytestmark = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "model.safetensors").exists(), reason="models/climatebert-tcfd not downloaded")


@pytest.fixture(scope="module")
def start():
    from wspr.start import load_start
    return load_start(DEFAULT_MODEL_DIR)


def test_param_arithmetic_matches_checkpoint(start):
    arch, weights, _ = start
    assert arch.full_params == 82_301_956
    assert S.count_params(weights, arch)["total"] == arch.full_params
    assert S.param_count(S.default_structure(arch), arch)["total"] == arch.full_params
    s = S.Structure((0, 1, 5), {0: tuple(range(1024)), 1: tuple(range(1024)), 5: tuple(range(1024))})
    assert S.param_count(s, arch)["total"] == 51_595_012
    pruned = S.prune_weights(weights, s, arch)
    assert S.count_params(pruned, arch)["total"] == 51_595_012
    assert S.structure_from_weights(pruned, arch) == (3, 1024)
    assert arch.param_count(4, 3072)["total"] == 68_126_212


def test_validate_rejections(start):
    arch = start[0]
    s = S.validate_structure({"layers": [0, 5], "ffn_keep": {"0": [1, 2], "5": [7, 9]}}, arch)
    assert s.layers == (0, 5) and s.m == 2 and not s.is_full(arch)
    bad = [
        ({"layers": []}, "non-empty"),
        ({"layers": [3, 1]}, "strictly increasing"),
        ({"layers": [0, 6]}, "within"),
        ({"layers": [0, 1], "ffn_keep": {"0": [1], "1": [1, 2]}}, "same number"),
        ({"layers": [0], "ffn_keep": {"0": [2, 1]}}, "sorted"),
        ({"layers": [0], "ffn_keep": {"0": [3072]}}, "within"),
        ({"layers": [0], "ffn_keep": {"3": [1]}}, "not kept"),
        ({"layers": [0], "ffn_keep": {"0": [1]}, "m": 5}, "does not match"),
    ]
    for d, msg in bad:
        with pytest.raises(S.StructureError, match=msg):
            S.validate_structure(d, arch)
    with pytest.raises(S.StructureError, match="twice"):
        S.prune_weights(S.prune_weights(start[1], s, arch), s, arch)


def test_prelude_param_count_agrees(start, tmp_path):
    from wspr.prelude import build_prelude
    arch = start[0]
    ns = {}
    exec(build_prelude(tmp_path, arch), ns)
    assert len(ns["catalog"]()) == len(start[1])
    assert all(not c["editable"] for c in ns["catalog"]() if c["name"].startswith("roberta.embeddings."))
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = int(rng.integers(1, 7)); m = int(rng.integers(1, 3073))
        layers = sorted(rng.choice(6, size=k, replace=False).tolist())
        out = ns["set_structure"](layers=layers, ffn_keep=m)
        assert "structure:" in out and "non-embedding" in out
        s = S.validate_structure(json.loads((tmp_path / "structure.json").read_text()), arch)
        assert ns["param_count"]()["total"] == S.param_count(s, arch)["total"]
    with pytest.raises(ValueError, match="frozen"):
        ns["save_edit"]("roberta.embeddings.word_embeddings.weight", ns["base"]("roberta.embeddings.word_embeddings.weight"))
    assert "cleared" in ns["clear_structure"]()


def test_truncated_model_matches_full_prefix(start):
    from wspr.forward import build_model, tokenizer
    arch, weights, _ = start
    full = build_model(weights, arch, torch.device("cpu"))
    s = S.Structure((0, 1), {0: tuple(range(3072)), 1: tuple(range(3072))})
    small = build_model(weights, arch, torch.device("cpu"), structure=s)
    assert small.config.num_hidden_layers == 2 and sum(p.numel() for p in small.parameters()) == S.param_count(s, arch)["total"]
    enc = tokenizer(arch)(["The board reviews climate-related risks twice a year.", "Scope 1 emissions fell 12%."], padding=True, return_tensors="pt")
    with torch.no_grad():
        hf = full(**enc, output_hidden_states=True).hidden_states
        hs = small(**enc, output_hidden_states=True).hidden_states
    assert len(hs) == 3
    for i in range(3):
        assert torch.allclose(hf[i], hs[i], atol=1e-5)


def test_export_loads_stock(start, tmp_path):
    from wspr.export import export_checkpoint
    from transformers import AutoModelForSequenceClassification, pipeline
    arch, weights, prov = start
    s = S.Structure((0, 1, 5), {l: tuple(range(1024)) for l in (0, 1, 5)})
    export_checkpoint(weights, tmp_path / "m", None, arch, structure=s)
    m = AutoModelForSequenceClassification.from_pretrained(tmp_path / "m")
    assert sum(p.numel() for p in m.parameters()) == 51_595_012 and m.config.id2label[0] == "governance"
    out = pipeline("text-classification", model=str(tmp_path / "m"), device=-1)("We disclose Scope 1 and 2 emissions annually.")
    assert out[0]["label"] in arch.labels

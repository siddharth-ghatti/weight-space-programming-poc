"""Unit tests for the pruning spec: validation, slicing, parameter arithmetic, equivalence."""
import json
import numpy as np
import pytest
import torch

from wsp.model import weights as W
from wspc import structure as S


def test_param_arithmetic_matches_checkpoint():
    full = S.default_structure()
    assert S.param_count(full)["total"] == S.FULL_PARAMS == 109_488_392
    base = W.load_base()
    assert S.count_params(base)["total"] == S.FULL_PARAMS
    s = S.Structure((0, 1, 11), {0: tuple(range(256)), 1: tuple(range(256)), 11: tuple(range(256))})
    assert S.param_count(s)["total"] == 32_712_968
    pruned = S.prune_weights(base, s)
    assert S.count_params(pruned)["total"] == 32_712_968
    assert S.structure_from_weights(pruned) == (3, 256)


def test_validate_rejections():
    ok = {"layers": [0, 5], "ffn_keep": {"0": [1, 2], "5": [7, 9]}}
    s = S.validate_structure(ok)
    assert s.layers == (0, 5) and s.m == 2 and not s.is_full()
    bad = [
        ({"layers": []}, "non-empty"),
        ({"layers": [3, 1]}, "strictly increasing"),
        ({"layers": [0, 12]}, "within"),
        ({"layers": [0, 1], "ffn_keep": {"0": [1], "1": [1, 2]}}, "same number"),
        ({"layers": [0], "ffn_keep": {"0": [2, 1]}}, "sorted"),
        ({"layers": [0], "ffn_keep": {"0": [1, 1]}}, "duplicate"),
        ({"layers": [0], "ffn_keep": {"0": [3072]}}, "within"),
        ({"layers": [0], "ffn_keep": {"3": [1]}}, "not kept"),
        ({"layers": [0, 1], "ffn_keep": {"0": [1]}}, "missing"),
        ({"layers": [0], "ffn_keep": {"0": [1]}, "m": 5}, "does not match"),
    ]
    for d, msg in bad:
        with pytest.raises(S.StructureError, match=msg):
            S.validate_structure(d)
    with pytest.raises(S.StructureError, match="twice"):
        S.prune_weights(S.prune_weights(W.load_base(), s), s)


def test_prelude_param_count_agrees(tmp_path):
    """The in-sandbox arithmetic and the harness arithmetic must agree."""
    from wspc.prelude import build_prelude
    src = build_prelude(tmp_path, W.BASE_DIR / "model.safetensors")
    ns = {}
    exec(src, ns)
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = int(rng.integers(1, 13)); m = int(rng.integers(1, 3073))
        layers = sorted(rng.choice(12, size=k, replace=False).tolist())
        out = ns["set_structure"](layers=layers, ffn_keep=m)
        assert "structure:" in out
        d = json.loads((tmp_path / "structure.json").read_text())
        s = S.validate_structure(d)
        assert ns["param_count"]()["total"] == S.param_count(s)["total"]
    # list / dict forms and the uniform-m check
    ns["set_structure"](layers=[2, 7], ffn_keep=[5, 1, 9])
    assert json.loads((tmp_path / "structure.json").read_text())["ffn_keep"]["7"] == [1, 5, 9]
    with pytest.raises(ValueError, match="same number"):
        ns["set_structure"](layers=[2, 7], ffn_keep={2: [1], 7: [1, 2]})
    assert "cleared" in ns["clear_structure"]()


def test_truncated_model_matches_full_prefix():
    """Keeping layers 0,1 at full width reproduces the first two layers of the 12-layer model."""
    from wspc.forward import build_model
    base = W.load_base()
    full = build_model(base, torch.device("cpu"))
    s = S.Structure((0, 1), {0: tuple(range(3072)), 1: tuple(range(3072))})
    small = build_model(base, torch.device("cpu"), structure=s)
    assert small.config.num_hidden_layers == 2 and sum(p.numel() for p in small.parameters()) == S.param_count(s)["total"]
    from wsp.model.forward import tokenizer
    enc = tokenizer()(["The premium is due monthly.", "Flood damage is excluded."], padding=True, return_tensors="pt")
    with torch.no_grad():
        hf = full(**enc, output_hidden_states=True).hidden_states
        hs = small(**enc, output_hidden_states=True).hidden_states
    assert len(hs) == 3
    for i in range(3):
        assert torch.allclose(hf[i], hs[i], atol=1e-5)

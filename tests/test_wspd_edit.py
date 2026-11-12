"""Unit tests of the distillation harness that need no model: the hooked column edit equals the baked weight, the
solve reproduces the normal equations, parsing and calibration helpers."""
import math

import numpy as np
import pytest
import torch

from wspd.common import ANSWER_WORDS, K, answer_words, ece, parse_word, prompt, repeat_run, selective_accuracy
from wspd.edit import ColumnEdit, smoothed_one_hot
from wspd.teacher import _fit_temperature


def test_prompt_and_parse():
    p = prompt("The insurer will not pay for flood damage.")
    assert p.endswith("Answer:") and "<think>" in p and all(w in p for w in ANSWER_WORDS)
    assert parse_word(" exclusion") == ANSWER_WORDS.index("exclusion")
    assert parse_word("<think>\n\n</think>\n\nAnswer: Claims procedure") == ANSWER_WORDS.index("claims")
    assert parse_word("claimsclaims") == ANSWER_WORDS.index("claims")
    assert parse_word("conditional") is None and parse_word("nothing here") is None
    assert parse_word(" coverage: States what") == 0
    assert answer_words("premium premium exclusion none") == 3 and answer_words("claimsclaims") == 1   # a run counts once
    assert repeat_run("a\na\na\nb") == 3


def test_column_edit_hook_equals_bake():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8, bias=False)
    lin.weight.data = lin.weight.data.half().float()      # exactly representable, so hook == bake to fp32 rounding
    x = torch.randn(5, 16)
    cols = [1, 4, 9]
    base_out = lin(x)
    e = ColumnEdit(None, 0, cols, module=lin)
    assert torch.allclose(lin(x), base_out)              # D = 0: the hook leaves the output unchanged
    e.D.copy_(torch.randn(8, 3) * 0.1)
    e.alpha = 0.0
    assert torch.allclose(lin(x), base_out)              # dose 0 leaves the output unchanged
    e.alpha = 1.0
    hooked = lin(x)
    info = e.bake()
    assert info["weights_in_edit"] == 24 and info["weights_changed"] == 24 and not lin._forward_hooks
    assert torch.allclose(lin(x), hooked, atol=1e-5)
    e.restore(info["backup"])
    assert torch.allclose(lin(x), base_out)


def test_normal_equations_solve():
    """dW = R^T K (K^T K + lam n C)^-1 minimises ||K dW^T - R||^2 + lam n tr(dW C dW^T): the gradient is zero there."""
    torch.manual_seed(1)
    n, I, d, lam = 40, 12, 6, 0.5
    Kmat, R = torch.randn(n, I, dtype=torch.float64), torch.randn(n, d, dtype=torch.float64)
    Z = torch.randn(200, I, dtype=torch.float64); C = Z.T @ Z / 200
    A = Kmat.T @ Kmat + lam * n * C
    dW = torch.linalg.solve(A, Kmat.T @ R).T
    grad = 2 * (Kmat @ dW.T - R).T @ Kmat + 2 * lam * n * dW @ C
    assert grad.abs().max() < 1e-8


def test_calibration_helpers():
    y = torch.tensor([0, 1, 2, 3])
    p = torch.eye(4)
    assert ece(p, y) < 1e-6
    sel = selective_accuracy(p, y, 0.5)
    assert sel["coverage"] == 0.5 and sel["accuracy"] == 1.0
    v = smoothed_one_hot(3)
    assert len(v) == K and abs(sum(v) - 1) < 1e-9 and v[3] == max(v)
    logits = np.array([[4.0, 0, 0, 0], [0, 4.0, 0, 0], [0, 0, 4.0, 0], [0, 0, 0, 4.0]] * 10)
    yy = np.array([0, 1, 2, 3] * 10)
    assert _fit_temperature(logits, yy) < 0.5           # confident and right: the fit sharpens
    logits2 = logits.copy(); yy2 = yy.copy(); yy2[::2] = (yy2[::2] + 1) % 4
    assert _fit_temperature(logits2, yy2) > _fit_temperature(logits, yy)   # half wrong: the fit softens

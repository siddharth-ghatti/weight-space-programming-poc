"""Integration test of the distillation harness on a few rows with a fake teacher (no API, needs the reader).

Run: uv run python -m pytest -q tests/test_wspd_offline.py
"""
import json

import numpy as np
import pytest
import torch

from wspd.common import K, MODEL_DIR, prompt
from wspd.data import load_rows, split_practice
from wspd.edit import Bench, DirectEditor, smoothed_one_hot
from wspd.eval.evaluate import score_at_answer_position, score_generation
from wspd.export import save_edit_bundle

pytestmark = pytest.mark.skipif(not list(MODEL_DIR.glob("*.safetensors")), reason="models/qwen3.5-0.8b not downloaded")


@pytest.fixture(scope="module")
def student():
    from wspd.reader import Student
    return Student(MODEL_DIR)


def test_one_round_edit_and_bundle(student, tmp_path):
    practice, test = load_rows()
    train, val = split_practice(practice, 40, 0)
    splits = {"train": train[:24], "val": val[:8], "test": test[:8]}
    for rows in splits.values():
        for r in rows:
            r["teacher"] = smoothed_one_hot(r["gold"])
    b = Bench(student, splits, chunk=8)
    assert b.spelling["used"] == "space" and b.n == {"train": 24, "val": 8, "test": 8}
    before = b.evaluate(("val",), forgetting=False)
    ed = DirectEditor(b, 23, 64, 1.0)
    assert len(ed.cols) == 64 and ed.A.shape == (64, 64)
    r = ed.run(1, [0.5, 1.0], 1.0, 0.05, eval_splits=("val",), forgetting=False)
    assert len(r["hist"]) == 2 and r["hist"][1]["solve_ms"] < 1000
    after = b.evaluate(("val",), forgetting=False)
    assert after["val_gap"] <= before["val_gap"] + 1e-6           # the kept round never makes the kept-back rows worse
    hooked = b.evaluate(("test",), forgetting=False)["test_gap"]
    info = ed.edit.bake()
    assert info["weights_in_edit"] == 1024 * 64
    assert abs(b.evaluate(("test",), forgetting=False)["test_gap"] - hooked) < 0.02   # baked == hooked up to fp16 rounding
    man = save_edit_bundle(tmp_path / "edit", 23, ed.cols, ed.edit.D, student.provenance, {"weights_changed": info["weights_changed"]})
    z = np.load(tmp_path / "edit" / "edit.npz")
    assert z["D"].shape == (1024, 64) and man["n_columns"] == 64 and len(man["D_sha256"]) == 64
    sc = score_at_answer_position(b, "test")
    assert set(sc) >= {"accuracy", "macro_f1", "ece", "selective", "agree_teacher"}
    g = b.generation_test(splits["test"], 4)
    s = score_generation([r["gold"] for r in splits["test"]], g)
    assert 0 <= s["accuracy"] <= 1 and s["n"] == 8
    ed.edit.restore(info["backup"])

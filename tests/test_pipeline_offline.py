"""End-to-end plumbing test with a scripted fake editor (no API calls).

Run: uv run pytest -q tests/test_pipeline_offline.py
"""
import json, shutil
from pathlib import Path
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from wsp.harness import driver as D
from wsp.harness.driver import Driver, load_config
from wsp.eval.evaluate import evaluate_run
from wsp.harness.audit import audit_run

SCRIPT = [
    ("text", "Let me look at the catalog first."),
    ("run_python", {"code": "c=catalog(); print(len(c)); print(W('bert.pooler.dense.weight').shape); print(len(practice()))"}),
    ("run_python", {"code": "import torch\nprint('should be rejected')"}),
    ("forward_pass", {"kind": "read", "model": "base", "texts": ["The premium is due monthly."], "layers": [12], "outputs": ["pooled"]}),  # free texts -> refused, must not count
    ("forward_pass", {"kind": "read", "model": "base", "ids": ["p0000", "p0001"], "layers": [12], "outputs": ["pooled", "cls_hidden", "logits", "attention"]}),
    ("run_python", {"code": "import numpy as np\np=np.load('pass_1/pooled.npy'); print(p.shape)\nH=np.zeros((NUM_LABELS,HIDDEN),dtype=np.float32); H[2]=p[0]; H[1]=p[1]\nprint(save_edit('classifier.weight',H)); print(save_edit('classifier.bias', np.arange(8,dtype=np.float32)*0.01)); open('notes.md','w').write('wrote a head from two pooled vectors')"}),
    ("forward_pass", {"kind": "predict", "model": "edited", "n": 9999}),  # invalid -> must not count
    ("forward_pass", {"kind": "predict", "model": "both", "n": 64}),
    ("forward_pass", {"kind": "predict", "model": "edited", "n": 32}),   # budget (2) exhausted -> refused
    ("submit", {"final_answer": "Set two rows of the head from pooled vectors."}),
]


class FakeStream:
    def __init__(self, msg): self.msg = msg
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self): return self.msg


class FakeMessages:
    def __init__(self): self.i = 0; self.seen = []
    def stream(self, **kwargs):
        self.seen.append(kwargs)
        kind, payload = SCRIPT[self.i]; self.i += 1
        if kind == "text":
            content = [TextBlock(type="text", text=payload)]; stop = "end_turn"
        else:
            content = [TextBlock(type="text", text=f"calling {kind}"), ToolUseBlock(type="tool_use", id=f"tu_{self.i}", name=kind, input=payload)]; stop = "tool_use"
        msg = Message(id=f"msg_{self.i}", type="message", role="assistant", model="fake", content=content, stop_reason=stop,
                      stop_sequence=None, usage=Usage(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=200, cache_read_input_tokens=5000))
        return FakeStream(msg)


class FakeClient:
    def __init__(self, **kw): self.messages = FakeMessages()


@pytest.fixture
def run_dir():
    d = D.RUNS / "_test_offline"
    shutil.rmtree(d, ignore_errors=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_offline_loop(run_dir, monkeypatch):
    monkeypatch.setattr(D.anthropic, "Anthropic", FakeClient)
    cfg = load_config(passes=2, max_turns=20, handoff_every=6)
    cfg["dollar_cap"] = 150
    drv = Driver("_test_offline", cfg)
    st = drv.run()
    assert st.done and st.end_reason == "submitted"
    assert st.passes_used == 2 and st.passes_failed == 2
    assert st.lint_rejections_total == 1
    assert st.code_runs == 2
    assert st.conversation == 2  # handoff happened at turn 6
    assert (run_dir / "edit" / "edit.npz").exists() and (run_dir / "model" / "model.safetensors").exists()
    man = json.loads((run_dir / "edit" / "manifest.json").read_text())
    assert {t["name"] for t in man["tensors"]} == {"classifier.weight", "classifier.bias"}
    # handoff prompt carried the notes
    hp = (run_dir / "turns" / "t007" / "prompt.txt").read_text()
    assert "wrote a head from two pooled vectors" in hp and "Forward passes left: **1**" in hp
    # cost accounting: 10 turns * (1000*4 + 500*20 + 200*5 + 5000*0.2)/1e6
    assert abs(st.cost_usd - 10 * (4000 + 10000 + 1000 + 1000) / 1e6) < 1e-9
    s = evaluate_run(run_dir)
    assert 0 <= s["test"]["accuracy"] <= 1 and "per_class" in s["test"]
    assert (run_dir / "eval" / "report.md").exists()
    a = audit_run(run_dir)
    assert a["code_runs"] == 2 and any(f["rejected"] for f in a["flagged_calls"])
    # the exported checkpoint loads with stock transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    m = AutoModelForSequenceClassification.from_pretrained(run_dir / "model")
    assert m.config.num_labels == 8 and m.config.id2label[1] == "exclusion"
    AutoTokenizer.from_pretrained(run_dir / "model")

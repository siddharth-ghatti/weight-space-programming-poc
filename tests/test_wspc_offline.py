"""End-to-end plumbing test of the pruning harness with a scripted fake editor (no API calls).

Run: WSP_FAST_EVAL=1 uv run python -m pytest -q tests/test_wspc_offline.py
"""
import json, os, shutil
from pathlib import Path
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from wsp.model import weights as W
from wspc import driver as D
from wspc.driver import Driver, load_config
from wspc.eval.evaluate import evaluate_run
from wspc.audit import audit_run
from wspc.export import export_from_edit_dir

INIT = W.ROOT / "runs" / "v1" / "p50"
pytestmark = pytest.mark.skipif(not (INIT / "edit" / "edit.npz").exists(), reason="runs/v1/p50 not present")

SCRIPT = [
    ("text", "Let me look at the start model and the structure."),
    ("run_python", {"code": "s=structure(); print(s['m'], len(s['layers']), len(catalog())); print(bool(base('classifier.weight').any())); "
                            "print(set_structure(layers=[0,1,11], ffn_keep=256)); print(param_count()['total'])"}),
    ("forward_pass", {"kind": "read", "model": "edited", "ids": ["p0000", "p0001"], "layers": [12], "outputs": ["cls_hidden"]}),  # refused: 0..3
    ("forward_pass", {"kind": "read", "model": "edited", "ids": ["p0000", "p0001"], "layers": [0, 3], "outputs": ["cls_hidden", "mean_hidden", "attention", "pooled"]}),
    ("forward_pass", {"kind": "predict", "model": "both", "n": 32}),
    ("run_python", {"code": "import numpy as np\nopen('notes.md','w').write('kept layers 0,1,11 at width 256')\n"
                            "b=W('classifier.bias'); b[0]+=0.01; print(save_edit('classifier.bias', b)); print(edits())"}),
    ("run_python", {"code": "open('structure.json','w').write('{\"layers\": [5, 2]}')"}),   # make it invalid
    ("submit", {"final_answer": "try to submit with a broken structure"}),                 # refused
    ("run_python", {"code": "print(set_structure(layers=[0,1,11], ffn_keep=256))"}),
    ("submit", {"final_answer": "Kept layers 0, 1 and 11 at FFN width 256; nudged the classifier bias."}),
]


class FakeStream:
    def __init__(self, msg): self.msg = msg
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self): return self.msg


class FakeMessages:
    def __init__(self): self.i = 0
    def stream(self, **kwargs):
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
    d = D.RUNS / "_test_wspc_offline"
    shutil.rmtree(d, ignore_errors=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_offline_prune_loop(run_dir, monkeypatch):
    monkeypatch.setattr(D.anthropic, "Anthropic", FakeClient)
    monkeypatch.setenv("WSP_FAST_EVAL", "1")
    cfg = load_config(passes=3, max_turns=20, handoff_every=6)
    drv = Driver("_test_wspc_offline", cfg)
    st = drv.run()
    assert st.done and st.end_reason == "submitted"
    assert st.passes_used == 2 and st.passes_failed == 1
    assert st.code_runs == 4 and st.conversation == 2
    # the refused read did not count and named the valid range
    calls = [json.loads(l) for l in (run_dir / "tool_calls.jsonl").read_text().splitlines()]
    fp = [c for c in calls if c["tool"] == "forward_pass"]
    assert "does not count" in fp[0]["result"] and "0..3" in fp[0]["result"]
    pred = json.loads(fp[2]["result"])
    assert pred["structure"]["params"] == 32_712_968 and pred["structure"]["layers"] == [0, 1, 11]
    read = json.loads(fp[1]["result"])
    assert read["layer_map"] == {"1": 0, "2": 1, "3": 11} and read["shapes"]["attention_L3"][1] == 12
    # the submit with a broken structure was refused
    subs = [c for c in calls if c["tool"] == "submit"]
    assert "Not submitted" in subs[0]["result"] and "strictly increasing" in subs[0]["result"]
    # handoff prompt carried notes and structure (handoff at turn 6 -> t007 prompt)
    hp = (run_dir / "turns" / "t007" / "prompt.txt").read_text()
    assert "kept layers 0,1,11 at width 256" in hp and "layers kept = [0, 1, 11]" in hp and "m = 256" in hp
    # export: 3 layers / 256 width, loads stock, param count exact, provenance recorded
    cfgj = json.loads((run_dir / "model" / "config.json").read_text())
    assert cfgj["num_hidden_layers"] == 3 and cfgj["intermediate_size"] == 256
    man = json.loads((run_dir / "edit" / "manifest.json").read_text())
    assert man["init_from"] == "runs/v1/p50" and man["exported_params"] == 32_712_968
    assert [t["name"] for t in man["tensors"]] == ["classifier.bias"]
    assert (run_dir / "edit" / "structure.json").exists()
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(run_dir / "model")
    assert sum(p.numel() for p in m.parameters()) == 32_712_968 and m.config.id2label[1] == "exclusion"
    # rebuild from public base + v1/p50 edit + this edit + structure is bit-identical
    rebuilt = run_dir / "rebuilt"
    export_from_edit_dir(run_dir / "edit", rebuilt)
    from safetensors import safe_open
    def hashes(d):
        with safe_open(str(d / "model.safetensors"), "np") as f:
            return {k: W.sha256(f.get_tensor(k)) for k in f.keys()}
    assert hashes(rebuilt) == hashes(run_dir / "model")
    # eval + audit
    s = evaluate_run(run_dir)
    assert s["model"]["total_params"] == 32_712_968 and "latency" in s and "meets_target" in s
    assert s["accuracy_target"] == 0.978
    assert "Held-out test accuracy" in (run_dir / "model" / "README.md").read_text()
    a = audit_run(run_dir)
    assert a["exported_params"] == 32_712_968 and a["structure_history"][0]["layers"] == [0, 1, 11]
    assert a["forward_pass_calls_with_free_texts"] == []

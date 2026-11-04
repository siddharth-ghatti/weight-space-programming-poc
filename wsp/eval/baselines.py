"""Reference models: unedited (zero head), linear probe on pooled features, full fine-tune.

Each is exported as a checkpoint directory under runs/_baselines/<name>/model and scored with
wsp.eval.evaluate so the numbers are directly comparable with editor runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wsp.labels import LABEL2ID, NUM_LABELS
from wsp.model import weights as W
from wsp.model.export import export_checkpoint
from wsp.model.forward import build_model, run_batches, pick_device, tokenizer, MAX_LEN
from wsp.eval.evaluate import DATA_DIR, evaluate_checkpoint, read_jsonl

BASELINES_DIR = W.ROOT / "runs" / "_baselines"


def baseline_unedited() -> dict:
    base = W.load_base()
    out = BASELINES_DIR / "unedited"
    export_checkpoint(base, out / "model", W.edit_manifest(base, {}))
    return evaluate_checkpoint(out / "model", out / "eval")


def baseline_probe(C: float = 1.0) -> dict:
    """Logistic regression on pooled [CLS] features of the practice set -> classifier head only.

    This is literally a weight edit of classifier.weight / classifier.bias, so it is also saved as an
    edit bundle. It answers: what does a gradient-free-looking but data-fitted head achieve?
    """
    from sklearn.linear_model import LogisticRegression

    base = W.load_base()
    practice = read_jsonl(DATA_DIR / "practice.jsonl")
    model = build_model(base)
    feats = run_batches(model, [r["text"] for r in practice], want_hidden=True)["pooled"]
    y = np.array([LABEL2ID[r["label"]] for r in practice])
    clf = LogisticRegression(C=C, max_iter=5000).fit(feats, y)
    edits = {
        "classifier.weight": clf.coef_.astype(np.float32),
        "classifier.bias": clf.intercept_.astype(np.float32),
    }
    out = BASELINES_DIR / "probe"
    manifest = W.save_edit_bundle(base, edits, out / "edit")
    export_checkpoint(W.apply_edits(base, edits), out / "model", manifest)
    return evaluate_checkpoint(out / "model", out / "eval", extra={"probe_C": C, "train_acc": float(clf.score(feats, y))})


def baseline_finetune(epochs: int = 3, lr: float = 2e-5, batch_size: int = 16, seed: int = 0) -> dict:
    """Standard full fine-tune with AdamW on the practice set (the ceiling)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = W.load_base()
    practice = read_jsonl(DATA_DIR / "practice.jsonl")
    device = pick_device()
    model = build_model(base, device)
    # Standard HF init for a fresh head (a zero head passes no gradient to the encoder).
    with torch.no_grad():
        model.classifier.weight.normal_(0.0, 0.02)
        model.classifier.bias.zero_()
    model.train()
    tok = tokenizer()
    texts = [r["text"] for r in practice]
    labels = torch.tensor([LABEL2ID[r["label"]] for r in practice])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(texts)
    steps_total = epochs * ((n + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps_total))
    log = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            y = labels[idx].to(device)
            out = model(**enc, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            loss_sum += out.loss.item() * len(idx); tot += len(idx)
            correct += int((out.logits.argmax(-1) == y).sum())
        log.append({"epoch": ep + 1, "loss": loss_sum / tot, "train_acc": correct / tot})
        print(log[-1])
    model.eval()
    weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items() if "position_ids" not in k}
    # keep the readable cls.* tensors so the dict shape matches base
    for k in base:
        if W.is_extra(k):
            weights[k] = base[k]
    out = BASELINES_DIR / "finetune"
    manifest = W.edit_manifest(base, {k: v for k, v in weights.items() if not W.is_extra(k)})
    export_checkpoint(weights, out / "model", manifest)
    return evaluate_checkpoint(out / "model", out / "eval", extra={"train_log": log, "epochs": epochs, "lr": lr})


def run_all() -> dict[str, dict]:
    results = {}
    for name, fn in (("unedited", baseline_unedited), ("probe", baseline_probe), ("finetune", baseline_finetune)):
        print(f"== baseline: {name}")
        s = fn()
        results[name] = {"accuracy": s["test"]["accuracy"], "macro_f1": s["test"]["macro_f1"]}
        print(json.dumps(results[name]))
    (BASELINES_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    return results

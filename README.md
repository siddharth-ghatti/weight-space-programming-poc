# Weight-Space Programming POC: insurance sentence classifier

An editor LLM (Claude Opus 5.5) is given the raw weights of `bert-base-uncased` in a Python sandbox
and must turn it into an 8-label insurance sentence classifier **without gradient descent**, under a
forward-pass budget. This replicates the setup of
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)), Experiment 8
style, on a new task.

An executive summary for non-technical readers is in `docs/executive-summary.md`.

## Setup

```bash
uv sync --extra dev                       # main venv (torch, transformers, anthropic)
uv venv .sandbox-venv --python 3.12       # restricted venv for the editor's code
uv pip install --python .sandbox-venv/bin/python numpy scipy safetensors tokenizers
uv run python -c "from huggingface_hub import snapshot_download as s; s('bert-base-uncased', local_dir='models/bert-base-uncased', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'])"
ant auth login                            # or export ANTHROPIC_API_KEY
```

## Pipeline

```bash
uv run python scripts/make_data.py               # synthesize data/practice.jsonl + data/test.jsonl (~$5-10)
uv run python scripts/run_baselines.py           # unedited / linear probe / full fine-tune -> runs/_baselines/
uv run python scripts/run.py --name smoke --passes 2 --max-turns 8 --model claude-sonnet-5-5   # plumbing check
uv run python scripts/run.py --name v1/p10 --passes 10  # real runs: v1/p00, v1/p10, v1/p25, v1/p50
uv run python scripts/evaluate.py runs/v1/p10    # (run.py does this automatically)
uv run python scripts/audit.py runs/v1/p10       # static audit of the editor's code
uv run python scripts/compare_runs.py            # runs/compare.md: one table across runs + baselines
uv run python scripts/predict.py --model runs/v1/p10/model "Flood damage is not covered under this policy."
uv run python scripts/scrub_paths.py                   # before publishing: replace absolute paths in logs with <repo>/<home>
uv run python scripts/push_hf.py runs/v1/p25/model --repo <user>/<name>   # push a checkpoint to the Hub
uv run python scripts/push_all_hf.py --user <hf-user> --private           # push every checkpoint, one repo each
```

## How it works

- **Sandbox** (`wsp/harness/sandbox.py`): each code run is a fresh `python -I` process from
  `.sandbox-venv` (numpy/scipy/safetensors/tokenizers; no torch/transformers), cleared env, cwd = the
  run's persistent `work/` folder, 900 s timeout, capped output, `sandbox-exec` profile that denies
  network and writes outside `work/`.
- **Prelude** (`wsp/harness/prelude.py`): `catalog()`, `base(name)`, `W(name)`, `save_edit(name, arr)`,
  `practice()`, `tokenize()`, `vocab()`. `save_edit` refuses frozen tensors (`bert.embeddings.*`),
  shape mismatches and non-finite values. No helper runs the model.
- **Forward passes** (`wsp/model/forward.py`) run in the harness, outside the sandbox, on MPS/CPU.
  `predict` scores base/edited/both on practice sentences; `read` writes hidden states, pooled output,
  logits or attention as `.npy` into `work/pass_<k>/`. Each successful pass costs one budget unit.
- **Driver** (`wsp/harness/driver.py`): manual tool loop over `run_python` / `forward_pass` / `submit`
  with the Anthropic SDK (adaptive thinking, effort knob, prompt caching, append-only history).
  Enforces the pass budget, 300 code runs, max turns, a hard dollar cap ($150 default), a static lint
  (torch/transformers/autodiff/sklearn/scipy.optimize rejected, returned unrun), and a fresh
  conversation with a handoff every 40 turns. Logs per turn under `runs/<name>/turns/tNNN/`.
- **Export** (`wsp/model/export.py`): `submit` writes `edit/edit.npz` + `manifest.json` (exact diff
  with sha256 of base and edited tensors) and `model/`, a stock `BertForSequenceClassification`
  checkpoint. `scripts/apply_edit.py` rebuilds `model/` bit-for-bit from the public base weights.
- **Evals** (`wsp/eval/evaluate.py`): accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix on
  the held-out test set, plus practice-set metrics (generalization gap). The test-set hash is checked
  before scoring. Baselines are exported and scored by the same code.

## Rules given to the editor

Same as the paper: no backward passes, optimizers or iterative fitting; no running the model inside
the sandbox; closed-form math on weights and on activations returned by forward passes is allowed.
Embeddings are frozen; encoder, pooler and the zero-initialized classifier head are editable.
From v1 on, rule 6: no self-authored data (practice set only; forward passes on practice ids only).

## Results

Editor: `claude-opus-5-5`, effort `xhigh`, one seed per budget. Test set: 1,779 held-out sentences
(never shown to the editor). Full table across all versions: `runs/compare.md` (from
`scripts/compare_runs.py`); per-run detail in `runs/<version>/<name>/eval/report.md`, `audit.json`,
`final_answer.md`, `work/notes.md`. Every run's `model/` folder is a complete Hugging Face checkpoint
with a model card; push one with `scripts/push_hf.py runs/v1/p25/model --repo <user>/<name>`.

### v0 (2026-11-03): paper rules only

Archived under `runs/v0/`. Rules 1-5 below; the editor was free to write its own texts and request
forward passes on them.

| run | passes used | turns | code runs | cost | test acc | test macro-F1 | tensors edited | model changed |
|---|---|---|---|---|---|---|---|---|
| unedited (zero head) | - | - | - | - | 13.1% | 0.029 | 0 | 0% |
| linear probe (logreg on pooled) | - | - | - | - | 88.5% | 0.887 | 2 | 0.006% |
| full fine-tune (3 ep, AdamW) | - | - | - | - | 97.3% | 0.973 | 201 | 100% |
| p00 | 0/0 | 28 | 27 | $2.65 | 98.0% | 0.980 | 108 | 36.7% |
| p10 | 6/10 | 37 | 30 | $3.55 | 96.7% | 0.967 | 40 | 12.4% |
| p25 | 18/25 | 187 | 170 | $14.02 | **98.4%** | **0.985** | 18 | 6.5% |
| p50 | 9/50 | 38 | 30 | $2.93 | 97.3% | 0.973 | 18 | 6.5% |
| smoke (Sonnet 5.5, 2 passes, 8 turns) | 1/2 | 8 | 6 | $0.26 | 87.9% | 0.881 | 2 | 0.006% |

Every editor run beats the linear probe, and three of four match or beat full fine-tuning. Practice
accuracy is 100% for p00/p25/p50 (99.7% for p10), so the practice-to-test gap is 1.6 to 3.3 points.
No run was stopped by the dollar cap, the turn limit, or the code-run limit; all four ended with `submit`.

### What each editor built

- **p00 (no forward passes): a bag-of-wordpieces classifier compiled into the encoder.** It never ran
  BERT. It tokenized the practice set, fit a naive-Bayes plus ridge keyword model in closed form, and
  wrote it into the weights: layer-0 FFN units detect keywords and write class scores into 8 residual
  directions, layer-1 attention averages them over the sentence, layers 2 to 11 are rewired as
  no-ops, and the pooler and head read the 8 scores out. The pretrained encoder is discarded.
- **p10: mean-pooling at layer 8, linear head.** Layer 8 attention is rewritten so [CLS] receives the
  mean of the layer-8 token states, layers 9 to 11 are made pass-through, the pooler standardizes the
  features, and the head is a ridge plus shrinkage-LDA fit on pooled features read back with a pass.
- **p50: kernel classifier in the layer-11 FFN.** Layers 0 to 10 untouched. Layer-11 attention is
  turned into uniform (mean) pooling, and the 3,072-unit FFN becomes 1,496 GELU units centered on the
  practice sentences' layer-11 features plus 16 linear units, with a least-squares readout. Only 9 of
  50 passes were used.
- **p25: same kernel design as p50, plus self-authored training data.** After reaching about 98%
  cross-validated accuracy on practice data alone, the editor wrote 2,201 labeled insurance sentences
  of its own in five batches (the last 255 aimed at the confusable label pairs it found in its
  cross-validation errors), spent passes reading their layer-11 features, and refit the kernel with
  3,072 centers (all 1,496 practice sentences plus 1,576 of its own) and a weighted ridge readout. It
  verified each version by predicting on 499 practice sentences and matching logits to its offline
  computation (max error 1.4e-4). Five conversations (four handoffs) over 187 turns.

### Observations

- **Forward passes are not the bottleneck.** p50 used 9 of 50; p10 used 6 of 10. The best run, p25,
  used passes for one purpose: reading layer-11 features of sentences (the practice set and its own)
  so the kernel could be fit offline, then one verification pass per saved version.
- **Closed-form fitting dominates.** Under the paper's rules (no backprop, no iterative fitting, closed-form
  math allowed), every editor fit its head with ridge regression, LDA, or naive Bayes on practice
  activations or tokens. The kernel-in-FFN design (p25, p50) is a closed-form kernel machine written
  into the network. The audit found no in-kernel reimplementation of the forward pass.
- **The pretrained encoder is optional for this task.** p00 reached 98.0% without using BERT at all,
  which says the synthetic data is largely separable by vocabulary. p25's extra 0.4 points come from
  the editor's own labeled sentences and a nonlinear readout, not from more passes.
- **The editor used its own knowledge as a data source.** p25's 2,201 self-written sentences were not
  prohibited by the rules. They are an LLM-specific advantage over a probe or fine-tune on the same
  practice set, and they should be considered when comparing against the baselines.
- **Lint false positives.** Three p25 code runs were rejected because the word "requests" appeared in
  the editor's sentence data (the network pattern matched prose). The editor rewrote and continued.
  The pattern now matches only imports and attribute calls; the historical rejections remain in
  `tool_calls.jsonl` and `state.json`.
- **Cost.** The whole matrix cost about $23 in editor calls, far under the $150-per-run cap. Dataset
  synthesis was a few dollars more.

### v1 (2026-11-04): rule 6, no self-authored data

Same matrix (p00/p10/p25/p50, one seed each) under `runs/v1/`, with one added rule and matching
enforcement: the editor may not write its own sentences, paraphrases, or keyword lists and use them
to fit, calibrate, select, or check weights; `forward_pass` accepts practice ids only (the `texts`
field is removed from the tool schema and refused by the driver, config `allow_free_texts: false`).
The post-hoc audit flags code runs containing many prose string literals
(`possible_self_authored_data_turns`) and any forward-pass request with free texts.

| run | passes used | turns | code runs | cost | test acc | test macro-F1 | tensors edited | model changed | v0 same budget |
|---|---|---|---|---|---|---|---|---|---|
| v1/p00 | 0/0 | 16 | 16 | $1.34 | 96.0% | 0.961 | 106 | 34.5% | 98.0% |
| v1/p10 | 9/10 | 48 | 38 | $3.43 | 97.3% | 0.973 | 144 | 5.0% | 96.7% |
| v1/p25 | 15/25 | 90 | 78 | $8.64 | 98.2% | 0.982 | 176 | 10.4% | 98.4% |
| v1/p50 | 12/50 | 48 | 38 | $4.12 | **98.3%** | **0.983** | 132 | 5.3% | 97.3% |

Baselines unchanged: linear probe 88.5%, full fine-tune 97.3%. Matrix cost $17.53. All four runs
submitted on their own; no lint rejections; no free-text forward-pass requests; the three
self-authored-data flags (p00 t2, p25 t22, p50 t25) were all false positives on long code or notes
string literals, confirmed by reading them.

**What each editor built (practice data only)**

- **p00: mean of input embeddings, ridge head, compiled into layer 0.** Attention queries and keys are
  zeroed so every token attends uniformly; the value projection applies a ridge classifier fit on
  practice token embeddings (with the position-embedding subspace projected out); the score is written
  into 8 residual directions at a large scale and carried unchanged to the pooler. Never ran BERT and
  predicted its own score from cross-validation (95.8%, actual 96.0%).
- **p10: early side channel plus mean pooling at layer 11.** Eight low-impact layer-0 FFN neurons
  write an 8-dim projection of word-level features into 8 low-variance hidden dimensions; layers 1 to
  10 are edited only to stop reading from or writing into those dimensions; layer 11 attention is
  rewritten so [CLS] becomes the normalized mean of all token states; the pooler holds a ridge head
  fit on edited-model activations read back with passes.
- **p25: stash at layer 0, readout at layer 9.** One layer-0 head computes the sentence mean of the
  embeddings and stores an 8-number summary in spare dimensions, protected through layer 8; a
  uniform-attention head at layer 9 adds the layer-9 token mean into [CLS]; layers 10 and 11 are
  pass-through; one ridge regression (penalty chosen by repeated cross-validation) on layer-9 [CLS],
  layer-9 mean and the stash gives the head. Verified on 1,024 practice sentences that the model's
  logits match the offline computation (correlation above 0.9999).
- **p50: reserved channel with three carry heads.** Eight hidden dimensions are cleared at layer 1
  and protected through layers 2 to 10; one attention head each in layers 2, 6 and 10 is repurposed
  as a uniform-attention mean-pooler whose value and output weights are ridge scorers writing into
  the channel; layer 11 pools the late-layer mean into [CLS]; the pooler decodes channel plus [CLS]
  with a joint ridge fit (penalty 300) on real edited-model activations.

**Observations**

- **Rule 6 cost almost nothing.** The best v1 run (98.3%) is within 0.1 points of v0's best (98.4%),
  which had 2,201 self-written sentences. The editors replaced extra data with better use of the
  pretrained encoder: every v1 run with passes combined an early-layer bag-of-words signal with a
  mid- or late-layer mean-pooled representation, carried through a reserved residual channel.
- **The pretrained encoder matters once the data shortcut is gone.** v1/p00, which cannot read
  activations, dropped to 96.0%; every run that could read activations beat full fine-tuning.
- **Passes are still underused.** 9, 15 and 12 passes used out of 10, 25 and 50. Passes went to
  reading all-layer features of the practice set, reading edited-model activations to fit heads on
  real values, and verifying that logits match the offline construction.
- **Convergent design.** Three of four v1 runs independently invented the same mechanism: a reserved
  8-dimensional channel in the residual stream, uniform-attention heads as mean-poolers, and
  pass-through final layers. The same pattern appeared in v0/p10 and v0/p50.
- **Cheaper than v0.** $17.53 versus $23.41, mostly because no run spent turns authoring data.
- **Uncalibrated scores.** Every edited model predicts the right label with flat softmax
  probabilities (mean top probability about 0.17 for v1/p50) because the editors wired the readout
  at a small logit scale. Multiplying the logits by any constant changes no prediction (checked on
  the test set), so accuracy and F1 are unaffected, but the probabilities need a temperature fit
  before use as confidences. Median top-two logit margin is 0.31 (v1/p50) versus 3.07 for fine-tuning.

### Checkpoints on the Hugging Face Hub

One repo per run, named `sidman007/wsp-bert-insurance-<version>-<run>` (for example
[v1-p50](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p50), the best run) and
`sidman007/wsp-bert-insurance-baseline-<name>` for the three baselines. Each loads with the stock
`text-classification` pipeline; see the calibration note above. Full list with scores in
`docs/executive-summary.md`.

## Run layout

```
runs/<version>/<name>/            # v0 = paper rules; v1 = + rule 6
  config.json  instructions.md  state.json  messages.json  driver.log
  turns/tNNN/{prompt.txt, answer.md, events.jsonl, run.json}
  requests/fwd_<k>.json          # each forward-pass request + result
  tool_calls.jsonl  sandbox_runs.jsonl  explanations.md  final_answer.md
  work/                          # the editor's persistent folder (edits/, pass_k/, notes.md, ...)
  edit/{edit.npz, manifest.json} # exact diff
  model/                         # runnable HF checkpoint (+ README.md model card), kept for HF push
  eval/{scores.json, per_item.jsonl, report.md}
  audit.json
```

## License

Apache 2.0 (code, logs, data and exported checkpoints). See `LICENSE`.

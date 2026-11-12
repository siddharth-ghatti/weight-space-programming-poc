# Weight-Space Programming POC: insurance sentence classifier

An editor LLM (Claude Opus 5.5) is given the raw weights of `bert-base-uncased` in a Python sandbox
and must turn it into an 8-label insurance sentence classifier **without gradient descent**, under a
forward-pass budget. This replicates the setup of
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)), Experiment 8
style, on a new task.

An executive summary for non-technical readers is in `docs/executive-summary.md`.
A two-page executive memo on the feasibility of the pruning method (experiments c1 and r1) is in `docs/exec-memo-model-pruning.md`.

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
# pruning experiment (c1): independent harness in wspc/, start model = runs/v1/p50
uv run python scripts/compress/run_baselines.py --suite                 # truncate+fine-tune and naive layer-drop baselines -> runs/_baselines/compress/
uv run python scripts/compress/run_baselines.py --matched 8,1024 --epochs 10 --lr 5e-5   # size-matched conventional baseline
uv run python scripts/compress/run.py --name c1/p50 --passes 50          # real runs: c1/p10, c1/p25, c1/p50 (auto evaluate + audit)
uv run python scripts/compress/apply_edit.py runs/c1/p50/edit --out /tmp/rebuilt --check runs/c1/p50/model   # exact rebuild
uv run python scripts/compress/benchmark.py runs/c1/p50/model            # params + CPU latency of any checkpoint
uv run python scripts/compress/compare_runs.py                           # runs/c1/compare.md
# reinsurance experiment (r1): independent harness in wspr/, start model = ClimateBERT's TCFD classifier
uv run python scripts/reins/fetch_model.py                              # models/climatebert-tcfd/ and models/climatebert-base-f/ (pinned commits)
uv run python scripts/reins/make_data.py                                # data/tcfd/{practice,test}.jsonl from climatebert/tcfd_recommendations
uv run python scripts/reins/measure_start.py                            # score the start model; writes the two targets into configs/reins.yaml
uv run python scripts/reins/run_baselines.py --suite                    # naive drops, truncate+fine-tune from the MLM base and from the task model
uv run python scripts/reins/run.py --name r1/p50 --passes 50            # real runs: r1/p25, r1/p50 (auto evaluate + audit)
uv run python scripts/reins/apply_edit.py runs/r1/p50/edit --out /tmp/rebuilt --check runs/r1/p50/model
uv run python scripts/reins/compare_runs.py                             # runs/r1/compare.md
uv run python scripts/reins/push_hf.py runs/r1/p50/model --repo <user>/<name>   # push an r1 checkpoint (RoBERTa tokenizer files)
# distillation experiment (d1): independent harness in wspd/, Jev-like teacher -> closed-form edits into Qwen3.5-0.8B
uv run python scripts/distill/fetch_model.py                            # models/qwen3.5-0.8b/ (pinned commit)
uv run python scripts/distill/fit_teacher.py --seed 0                   # calibrated heads on the frozen reader -> runs/d1/teacher_s0/ (+ targets)
uv run python scripts/distill/run.py --seed 0 --export-model            # layer sweep, grid, direct edit, bake, test + forgetting -> runs/d1/direct_s0/
uv run python scripts/distill/run_lora.py --seed 0                      # the gradient-descent arm on the same loss and rows -> runs/d1/lora_s0/
uv run python scripts/distill/run_unedited_full_prompt.py               # reference row: the unedited reader with the long prompt
uv run python scripts/distill/run_ablations.py                          # random columns, no protection, hard labels, 25%/50% rows, random change
uv run python scripts/distill/check_rebuild.py runs/d1/direct_s0        # base + edit bundle == baked checkpoint, tensor by tensor
uv run python scripts/distill/compare.py                                # runs/d1/compare.md
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

## Pruning harness (`wspc/`)

A second, independent harness for the compression experiment (c1). It imports read-only utilities
from `wsp` (labels, data paths, base weights, sandbox, lint, scoring) and changes nothing there, so
v0/v1 reproduce exactly as they ran. Same methodology: the editor only reads and writes weights, under
a forward-pass budget, no gradient descent, rules 1 to 6 verbatim. The objective changes: start from
the best v1 model (`runs/v1/p50`, 98.31%, 109,488,392 parameters) and **minimize exported parameters
subject to held-out accuracy >= 97.8%** (within 0.5 points of the start, above the 97.3% fine-tune).

- **Structured reduction only.** Two levers, both expressible in a stock checkpoint: drop encoder
  layers (`num_hidden_layers`) and narrow every kept layer's feed-forward block to the same width
  (`intermediate_size`), keeping any subset of the 3,072 units per layer. Vocabulary and embeddings
  stay intact (23.8M parameters, the floor). Zeroed heads or weights earn no credit; transformers
  5.22.1 has no head pruning. `total = 24,433,928 + k · (2,366,208 + 1,537 · m)`.
- **Edits at full shape, structure applied by the harness.** `save_edit` works on full-size tensors
  as before; `set_structure(layers, ffn_keep)` writes `structure.json`, which the harness applies when
  it builds the edited model for a forward pass and at export (slice `intermediate.dense`,
  `output.dense` columns, renumber layers). `forward_pass` on `base` is the unpruned 12-layer start;
  on `edited` it is the pruned model, with a `layer_map` from pruned to original layer ids. An invalid
  structure fails a pass without charging it and makes `submit` refuse.
- **Export and provenance** (`wspc/export.py`, `wspc/bundle.py`): `edit/` holds the diff relative to
  the v1/p50 start plus `structure.json`; `model/` is a pruned `BertForSequenceClassification` with
  `structure.json`, `param_count.json` and a model card; `scripts/compress/apply_edit.py` rebuilds it
  from bert-base + the v1/p50 edit + the c1 edit + the structure and checks identity.
- **Evaluation** (`wspc/eval/`): the `wsp` scorer plus parameter counts, checkpoint size, CPU latency
  (4 threads, median ms/sentence at batch 1 and 32 over 64 test sentences) and a pass/fail against the
  target. Baselines (`wspc/eval/baselines.py`): truncate bert-base to k layers (optionally narrow the
  FFN by a magnitude heuristic) and fine-tune on the practice set; naive layer drops of v1/p50 with
  no refit; the unpruned start.

## Reinsurance pruning harness (`wspr/`)

A third independent harness, for the question a reinsurer actually faces: a classifier somebody else
trained, with public training data, that has to run over millions of disclosure paragraphs on CPU.
Start model: [climatebert/distilroberta-base-climate-tcfd](https://huggingface.co/climatebert/distilroberta-base-climate-tcfd)
(DistilRoBERTa, 6 layers, 82,301,956 parameters, Apache-2.0), which sorts paragraphs of annual and
sustainability reports into the four TCFD pillars (governance, strategy, risk management, metrics).
Data: [climatebert/tcfd_recommendations](https://huggingface.co/datasets/climatebert/tcfd_recommendations)
(CC BY-NC-SA 4.0, research use): the train split minus the "none" class is the editor's practice set
(1,000 paragraphs; it is also the start model's own fine-tuning data, and the prompt says so), the
test split minus "none" is the held-out set (320 paragraphs: strategy 197, metrics 49, risk 48,
governance 26). The start model scores 77.50% accuracy and 0.7096 macro-F1 there (its card reports
no metrics); the bar is **accuracy >= 76.50% and macro-F1 >= 0.6896** (start minus 1 point and
minus 0.02), the F1 condition guarding the 26-paragraph governance class.

- Same methodology and rules 1 to 6 as `wspc/`; the package is written against an `Arch` record
  (`wspr/arch.py`) read from the checkpoint config (tensor prefixes, head layout, parameter
  arithmetic), so nothing BERT-specific is inherited. Reused from `wsp`: the sandbox, the lint, the
  audit helpers. `wsp/` and `wspc/` are unchanged.
- RoBERTa differences the harness handles: no pooler (the head is `tanh(classifier.dense(h[<s>]))`
  then `classifier.out_proj`; a `pooled` read returns that vector), byte-level BPE tokenizer files
  copied unchanged, `position_ids` buffer excluded from edits, 512-token paragraph inputs
  (`MAX_LEN` 512, batch 16), per-class recall in every predict pass.
- Parameter arithmetic (verified against the checkpoint): `total = 39,774,724 + k · (2,366,208 +
  1,537 · m)`; the embeddings are 39,181,056 of the 82.3M (47.6%), so the write-up reports total and
  non-embedding reduction.
- Baselines (`wspr/eval/baselines.py`, under `runs/_baselines/reins/`): naive layer drops of the
  start model; truncate the MLM base `climatebert/distilroberta-base-climate-f` to k layers, new
  head, fine-tune on the practice paragraphs (3 ep / 2e-5 and 10 ep / 5e-5); truncate the fine-tuned
  task model itself and fine-tune (10 ep); a size-matched variant at the best run's (k, m).

## Distillation harness (`wspd/`)

A fourth independent package with no editor LLM in the loop. It replicates the method of the Kaggle notebook
[`nandakishorconvai/fast-edits-decision-model`](https://www.kaggle.com/code/nandakishorconvai/fast-edits-decision-model)
("policy distillation by direct weight edit, against LoRA") on the insurance benchmark: a skill is learned
cheaply on the side by a Jev-like decision model, then written into a base language model's weights by one
linear-algebra solve instead of training, and the result is compared with LoRA on the same loss and rows and
with the Opus-edited BERT of the first experiment.

- **Reader / student:** [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (24 layers, hidden 1024,
  FFN 3584, 248k vocabulary, Apache-2.0), frozen, fp16 on Apple Metal (`wspd/reader.py`). Prompted with a
  sentence, the question and eight options, it answers with one of eight single-token words (` coverage`,
  ` exclusion`, ` premium`, ` deductible`, ` claims`, ` definitions`, ` cancellation`, ` conditions`), read at
  the next-token position after the chat model's empty thinking block (`wspd/common.py`).
- **Teacher, the Jev-like model (`wspd/teacher.py`):** multinomial logistic heads on the frozen reader's
  residual stream at the answer position (and the mean over the sentence tokens) at a few layers, feature set
  and C chosen by 5-fold cross-validation on the 1,456 training rows, one temperature fitted on out-of-fold
  probabilities, confidence `1 - H/log 8`, abstain band at 80% coverage. It answers a typed choice question
  with calibrated probabilities and never generates text: the construction of a vegaml/Laya policy head, under
  our control. Its probabilities on the training rows are the targets every arm distils.
- **Direct edit (`wspd/edit.py`):** one round = (1) one forward pass over the training prompts reads the input
  of the chosen `mlp.down_proj` at the answer position (the keys K) and the final residual h; (2) the wanted
  logit change `d = log(teacher) - log(student)` on the eight answer words becomes the smallest residual change
  `dh = J^+ d` through the readout Jacobian (final norm and the eight unembedding rows only); (3) the column
  update is the ridge solution `dW_S = R^T K_S (K_S^T K_S + lambda n C_SS)^-1`, with C the covariance of the keys
  on the guard text, every other token of the prompts and the token after the answer (MEMIT's normal equations
  restricted to the columns S), solved in milliseconds; (4) a dose in {0.5, 1, 1.5, 2} is chosen on 40 kept-back
  practice rows (val gap + guard KL + post-answer KL), doses that move the guard text past 0.03 nats/token are
  refused. Rounds repeat until no dose helps (max 6); the best round on the kept-back rows is kept and baked
  into the fp16 weights. Columns are the 256 (or 512, 1024) most active at the answer position relative to
  elsewhere. Layer in {17, 19, 21, 23}, width and lambda in {0.3, 1, 3} are chosen on a 256-row subset of the
  training rows by agreement with the teacher on the kept-back rows, then the most protection within one row of
  the best.
- **LoRA arm (`wspd/lora.py`):** rank 16 on gate/up/down of all 24 layers, Adam 5e-4, 60 steps of 16 rows,
  the same loss (teacher gap + guard KL + post-answer KL), checkpoint = best on the same 40 rows, merged into
  the weights. Trained in fp32 (the backward through the gated-DeltaNet reference path is NaN in fp16 on
  Metal), then the baked weights are measured in the same fp16 reader as every other arm.
- **Measurement (`wspd/eval/`):** on the 1,779 held-out sentences, "generated" = the first word the detached
  model writes (4 greedy tokens, unanswered counts as wrong) and "scored" = argmax over the eight answer words
  at the answer position; accuracy, macro-F1 (`wsp.eval.evaluate.full_metrics`), ECE, selective accuracy at
  80% coverage, agreement with the teacher. Forgetting suite (`wspd/retention.py`): WikiText-2 perplexity and
  KL against the unedited model's top-64 (40 test paragraphs), 200 ARC-Easy test questions scored by the answer
  letter, 40 capital cities, and answer words or repeated lines leaking into 15 unrelated free generations.
- **Ablations:** random columns with the same solve; no protection (lambda 0, no cap, no guard); hard gold
  labels instead of the calibrated teacher; 25% and 50% of the training rows; a random change of the same size
  on the same columns.
- **Artifacts:** `runs/d1/<arm>/edit/manifest.json` (layer, columns, hash of the column update, base model
  commit; `edit.npz` holds the update itself), `eval/scores.json` + `report.md`, and for seed 0 the baked
  checkpoint under `model/` (loads with stock `AutoModelForCausalLM`; `check_rebuild.py` confirms base +
  bundle == baked tensor by tensor). Nothing in this experiment calls an API: $0.

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

### c1 (2026-11-04): parameter reduction by weight editing

Same methodology, new objective: start from `runs/v1/p50` (98.31%, 109,488,392 parameters) and make
it as small as possible while staying at or above 97.8% on the held-out test set, using only the
structured levers above (drop layers, narrow the FFN; vocabulary and embeddings intact). One seed per
setting, Opus 5.5, under `runs/c1/`. The conventional comparison is the standard compression recipe:
truncate bert-base to k layers (and optionally narrow the FFN by a magnitude heuristic), then fine-tune
on the same 1,496 practice sentences; shown at both the standard recipe (3 epochs, lr 2e-5) and a
stronger one (10 epochs, lr 5e-5).

| model | passes used | turns | code runs | cost | layers / FFN | params | vs start | test acc | macro-F1 | target |
|---|---|---|---|---|---|---|---|---|---|---|
| start (v1/p50) | - | - | - | - | 12 / 3072 | 109.5M | - | 98.31% | 0.983 | pass |
| c1/p10 | 8/10 | 77 | 70 | $7.59 | 10 / 3072 | 95.3M | -12.9% | 97.41% | 0.975 | fail |
| c1/p25 | 12/25 | 77 | 65 | $7.78 | 10 / 3072 | 95.3M | -12.9% | 98.26% | 0.983 | pass |
| **c1/p50** | 47/50 | 236 | 201 | $19.39 | **8 / 1024** | **56.0M** | **-48.9%** | **97.92%** | **0.979** | **pass** |
| truncate + fine-tune, 6 layers, 10 ep | - | - | - | local | 6 / 3072 | 67.0M | -38.8% | 98.20% | 0.982 | pass |
| truncate + fine-tune, 4 layers, 10 ep | - | - | - | local | 4 / 3072 | 52.8M | -51.8% | 97.58% | 0.976 | fail |
| truncate + fine-tune, 2 layers, 10 ep | - | - | - | local | 2 / 3072 | 38.6M | -64.7% | 96.80% | 0.968 | fail |
| **size-matched: truncate + FFN-narrow + fine-tune, 8 / 1024, 10 ep** | - | - | - | local | 8 / 1024 | 56.0M | -48.9% | **97.08%** | 0.971 | fail |
| truncate + fine-tune, 6 / 4 / 2 layers, 3 ep | - | - | - | local | | 67.0M / 52.8M / 38.6M | | 93.1% / 85.6% / 83.5% | | fail |
| naive drop of v1/p50 layers 3-5, 7-9 (no refit) | - | - | - | - | 6 / 3072 | 67.0M | -38.8% | 54.6% | 0.565 | fail |

CPU latency (Apple M-series, 4 threads, median ms per sentence over 64 test sentences, measured
sequentially with nothing else running) and checkpoint size:

| model | params | MB | ms/sentence, batch 1 | ms/sentence, batch 32 |
|---|---|---|---|---|
| start (v1/p50) | 109.5M | 438 | 34.8 | 14.8 |
| c1/p25 (10 layers) | 95.3M | 381 | 29.5 | 12.5 |
| **c1/p50 (8 layers, FFN 1024)** | **56.0M** | **224** | **15.8** | **5.8** |
| truncate + fine-tune, 6 layers | 67.0M | 268 | 17.9 | 7.6 |
| size-matched fine-tune, 8 / 1024 | 56.0M | 224 | 14.7 | 5.8 |

Full table with every baseline: `runs/c1/compare.md`.

Matrix cost $34.76 (plus $1.51 for a Sonnet smoke run). All three runs submitted on their own; no
lint rejections; no free-text forward-pass requests; no in-kernel forward passes. The one
self-authored-data flag (p50 t117) is a false positive on a notes file written through a
triple-quoted string. Every exported checkpoint rebuilds bit-for-bit from bert-base + the v1/p50
diff + the c1 diff + `structure.json` (`apply_edit.py --check`).

**What each editor built**

- **p10 and p25: drop two layers, rebuild layer 11 as a kernel classifier.** Both kept original layers
  0 to 8 unchanged and dropped 9 and 10. Both measured that kernel-ridge accuracy on the token-mean of
  each layer's hidden states jumps after layer 8 and flattens after, and both tested FFN narrowing
  (2,560 to 2,816 units in p10; 1,536 and 2,560 in p25) and rejected it because the deep features
  degraded by 0.3 to 0.8 points. Layer 11 became a mean-pooling layer (queries and keys zeroed so
  [CLS] receives the plain token average) whose FFN is an exact kernel machine: one unit per practice
  sentence computing GELU of a scaled inner product with that sentence's pooled vector, output weights
  equal to the closed-form kernel-ridge coefficients, written into the 8 reserved dimensions. p25 also
  re-solved the start model's carry heads in layers 2 and 6 and fit one joint kernel + linear model.
  p25 explicitly weighed a 9-layer variant (88.2M) and rejected a 14 to 19% estimated chance of
  missing the bar for a 7.5% saving. p10's 20-fold cross-validation estimate was 98.5%; it fit with a
  small ridge penalty on 100% of the practice set and landed at 97.41% on test, the only run to fail.
- **p50: narrow every layer, drop two more, re-fit layer by layer.** Kept original layers 0, 1, 2, 3,
  5, 7, 8 and 11 at 1,024 FFN units each. The 1,024 units with the most output energy survive in
  each content layer; each narrowed block's output weights were then refit by ridge regression to
  reproduce the original model's next hidden state, sequentially from the bottom so each refit sees
  the already-pruned layers below it (drift-corrected). Layers 3 and 5 were refit to produce the
  hidden state the original model had two layers later, standing in for dropped layers 4 and 6
  (cross-validation: 98.28% without layer 4, 98.24% with it). The start model's three carry heads were
  replaced by one: layer 0's near-uniform head 0 writes a linear class score from the embedding
  average into the 8 reserved dimensions, protected to the end. Layer 11 is a mean-pool plus kernel
  classifier whose 1,024 units hold 1,008 anchor sentences chosen by pivoted Cholesky (random
  anchors lost 0.4 points) and 16 linear pass-through units; the kernel and linear parts were fit
  jointly, which the editor's simulation showed was worth 0.25 points over fitting them separately.
  Harness logits on all 1,496 practice sentences matched the editor's numpy reconstruction to 5e-6.
  Tested and abandoned: a 1-layer design (estimated ceiling 97.7%), dropping layer 8 (its stand-in
  could not approximate it), FFN widths 768 and 1,536 (768 too lossy, 1,536 no better than 1,024).

**Observations**

- **Budget bought compression, not accuracy.** With 10 or 25 passes the editors took the safe 13%
  cut (two late layers) and stopped. With 50 passes the editor spent 47 of them re-fitting narrowed
  blocks one layer at a time and reached a 49% cut at the same accuracy bar. The extra $12 of API
  spend is what separates the two outcomes.
- **Weight editing beats the conventional recipe at matched size.** At exactly the p50 shape (8
  layers, FFN 1,024), truncating bert-base and fine-tuning for 10 epochs reaches 97.08%, 0.84 points
  below the edited model and below the bar. The conventional recipe needs 6 full-width layers (67.0M)
  to pass; the edited model passes at 56.0M.
- **The unpruned model's structure is not free to cut.** Naively dropping the six layers whose only
  edits were channel protection destroys the model (54.6%), because the later carry heads were fit
  on the 12-layer hidden states. Every successful run re-fit whatever came after a cut from reads of
  the pruned model.
- **Cross-validation on the practice set is an optimistic proxy.** p10 estimated 98.5% and scored
  97.41%; p50 estimated 98.2 to 98.3% and scored 97.92%; p25 estimated 98.1% and scored 98.26%. The
  kernel classifiers interpolate the practice set exactly (100% practice accuracy), so the margin to
  the bar has to come from the editor's own conservatism.
- **Same convergent design as v1.** All three runs turned the last layer into a mean-pooling layer
  with a kernel machine in its FFN, writing class scores into the reserved 8-dimensional channel and
  decoding them in the pooler. The editors never trained anything: every fit was a closed-form ridge,
  kernel ridge or least-squares solve on activations returned by forward passes.
- **Same calibration caveat as v1.** The edited models' logits live at a small scale (p50 logit
  range -0.46 to 0.93 on the practice set); predictions are unaffected but probabilities need a
  temperature fit before use as confidences.

### r1 (2026-11-05): shrinking a third-party classifier for a reinsurance use case

**Problem.** Reinsurers carry the climate exposure of the companies they reinsure (cedents) and of
large corporate risks in treaty and facultative programs, and report on that exposure themselves
(TCFD/ISSB, EU CSRD, NAIC climate survey). Underwriting and ESG teams read cedents' and insureds'
annual and sustainability reports to find climate-related passages and sort them by TCFD pillar, to
judge whether a counterparty's climate-risk management is substantive or "cheap talk". Over thousands
of counterparties and hundreds of pages each, that is a CPU batch job over millions of paragraphs,
preferably on-premises. ClimateBERT's TCFD classifier is the public model for the task. Question: given
a vendor-trained model and its public training data, can weight editing deliver a cheaper deployable
version at the same accuracy, with no training and a full audit trail? Same methodology, same rules,
the harness in `wspr/` (above), one seed per setting, 25 and 50 passes, Opus 5.5, under `runs/r1/`.

The conventional comparisons: naive single- and double-layer drops of the start model (no refit); the
standard compression recipe (truncate the MLM base `distilroberta-base-climate-f` to k layers, new
head, fine-tune on the 1,000 practice paragraphs, 3 ep / 2e-5 and 10 ep / 5e-5); the same starting
from the fine-tuned task model (10 ep); and size-matched variants at the editors' shape.

| model | passes used | turns | code runs | cost | layers / FFN | params | vs start | non-emb. | test acc | macro-F1 | gov. recall | target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start (ClimateBERT TCFD) | - | - | - | - | 6 / 3072 | 82.3M | - | - | 77.50% | 0.7096 | 0.731 | pass |
| r1/p25 | 21/25 | 118 | 99 | $9.56 | 6 / 1792 | 70.5M | -14.3% | -27.4% | 77.81% | 0.7104 | 0.731 | pass |
| **r1/p50** | 32/50 | 151 | 125 | $10.14 | **6 / 1536** | **68.1M** | **-17.2%** | **-32.9%** | **78.75%** | **0.7206** | 0.731 | **pass** |
| naive drop layer 4 (no refit) | - | - | - | - | 5 / 3072 | 75.2M | -8.6% | -16.4% | 78.75% | 0.7023 | 0.731 | pass |
| naive drop layer 1 (no refit) | - | - | - | - | 5 / 3072 | 75.2M | -8.6% | -16.4% | 76.56% | 0.6918 | 0.692 | pass |
| naive drop layers 1 and 4 | - | - | - | - | 4 / 3072 | 68.1M | -17.2% | -32.9% | 77.19% | 0.6851 | 0.654 | fail (F1) |
| every other naive drop (0, 2, 3 or 5; any other pair) | - | - | - | - | 5 or 4 / 3072 | | | | 61.6 to 76.6% | 0.19 to 0.68 | | fail |
| truncate MLM base + fine-tune, 4 / 3 / 2 layers, 10 ep | - | - | - | local | 4, 3, 2 / 3072 | 68.1 / 61.0 / 54.0M | | | 74.4 / 73.8 / 72.2% | 0.675 / 0.660 / 0.665 | | fail |
| truncate task model + fine-tune, 4 / 3 layers, 10 ep | - | - | - | local | 4, 3 / 3072 | 68.1 / 61.0M | | | 74.4 / 73.4% | 0.680 / 0.664 | | fail |
| truncate MLM base + fine-tune, 4 / 3 / 2 layers, 3 ep | - | - | - | local | | | | | 68.1 / 64.7 / 61.6% | 0.46 / 0.39 / 0.19 | | fail |
| **size-matched at p50's shape:** truncate task model, narrow FFN by magnitude, fine-tune 3 ep | - | - | - | local | 6 / 1536 | 68.1M | -17.2% | -32.9% | **75.62%** | 0.679 | 0.615 | fail |
| size-matched: same, 10 ep / 5e-5 | - | - | - | local | 6 / 1536 | 68.1M | -17.2% | -32.9% | 75.00% | 0.679 | 0.654 | fail |
| size-matched: from the MLM base, new head, 10 ep | - | - | - | local | 6 / 1536 | 68.1M | -17.2% | -32.9% | 72.19% | 0.641 | 0.615 | fail |

CPU latency (Apple M-series, 4 threads, median ms per paragraph over 64 test paragraphs at the full
512-token window, measured sequentially with nothing else running) and checkpoint size. Batch 16 is
slower per paragraph than batch 1 here because paragraphs vary from 30 to 500 tokens and a batch pads
to its longest member; a length-sorted batching loop would recover that in deployment.

| model | params | MB | ms/paragraph, batch 1 | ms/paragraph, batch 16 |
|---|---|---|---|---|
| start (ClimateBERT TCFD) | 82.3M | 329 | 25.2 | 30.1 |
| naive drop layer 4 | 75.2M | 301 | 21.0 | 25.3 |
| r1/p25 (6 layers, FFN 1792) | 70.5M | 282 | 18.7 | 20.8 |
| **r1/p50 (6 layers, FFN 1536)** | **68.1M** | **273** | **17.3** | **19.6** |
| size-matched fine-tune, 6 / 1536 | 68.1M | 273 | 17.2 | 19.7 |

Full table with every baseline: `runs/r1/compare.md`.

Matrix cost $19.70 (plus $0.93 for a Sonnet smoke run). Both runs submitted on their own; no
free-text forward-pass requests; no in-kernel forward passes. p50 had one lint rejection (a scipy
root-finder for a calibration threshold, returned unrun and rewritten) and two self-authored-data
flags (t49, t118) that are notes-file appends through triple-quoted strings. Both checkpoints
rebuild bit-for-bit from the pinned Hub checkpoint + the edit diff + `structure.json`.

**What each editor built**

- **Both: keep every layer, narrow every feed-forward block, re-solve its output projection.** Both
  runs measured that dropping any whole layer was much worse than narrowing (p25: agreement with the
  original's predictions fell to 94 to 95.5% even with the head refit), which the naive-drop baselines
  confirm: only layer 4 is free to drop, and no pair is. Both read the actual inputs of each FFN block
  on practice paragraphs (token-level for the early layers; only the `<s>` position for layer 5,
  since that is all the head reads), kept the units with the largest output-weight energy times
  activation variance, and refit each block's `output.dense.weight` and bias by ridge regression so
  the kept units reproduce the full block's output, dropped units included. Attention, LayerNorms,
  input projections and the classifier head were left alone. Edits: 12 tensors in each run.
- **p25 stopped at 1,792 units.** It judged 1,536 too risky: practice macro-F1 dropped 0.021, just
  over the 0.02 allowance, and prediction changes on borderline paragraphs were 28% larger. It
  estimated a 1-in-5 chance of missing the bar at 1,792 and landed above the start model on both
  metrics.
- **p50 went to 1,536 with a risk model.** It ranked layers 3 and 4's units by their effect after the
  following LayerNorm, fit the later layers on inputs from the already-pruned model, and simulated
  test sets by shrinking the original model's practice logits to a 77.5% accuracy and adding the
  logit noise each pruned build actually showed at similar margins. Estimated failure chance 13 to
  14% at 1,536 versus 7% at 2,048; it took the size. It also tried 1,024 units (sequential, layer by
  layer), head refits, greedy unit selection and logit corrections, and kept none. Result: 78.75% and
  0.7206, above the start model on both metrics and the best of every row in the table.

**Observations**

- **Weight editing beat every conventional option.** The strongest conventional result that holds
  the bar is dropping layer 4 with no refit (75.2M). The standard recipe, truncate and fine-tune,
  fails at every depth from either starting point: with 1,000 training paragraphs and a hard,
  imbalanced task it cannot recover the accuracy it throws away. The editors reached 68.1M (p50) and
  70.5M (p25) while improving accuracy and macro-F1 slightly.
- **Narrowing, not depth, was the lever here.** On a 6-layer model each layer is a sixth of the
  encoder, and both editors found every layer load-bearing. The feed-forward blocks were a third
  redundant: a third of the encoder parameters went with no measurable loss once the output
  projections were re-solved. The embeddings (47.6% of the model) are untouched by design, which
  caps the total reduction; the non-embedding reduction is the number that tracks CPU compute.
- **The practice set being the model's training data was not the problem it could have been.** The
  start model scores only 92.5% on its own training data, so the editors' fits were not fitting
  memorized labels, and both of their estimates were pessimistic rather than optimistic: p25
  expected 1-in-5 odds of a miss and gained 0.3 points; p50 expected 13% odds and gained 1.25
  points. That reverses the c1 pattern, where estimates ran optimistic.
- **Small test set, honest margins.** 320 paragraphs, 26 of them governance: one paragraph is 0.31
  accuracy points and one governance paragraph moves its recall by 3.8 points. Differences under a
  point in this table are within noise. Both editor results are above the start model, which is the
  reassuring reading; the ordering between them is not.
- **Cheaper than c1.** The whole matrix cost $19.70 for two runs; the editors used 21 and 32 passes.
  Paragraph inputs made every pass slower, but the runs still finished in 1 h 22 min and 1 h 37 min.
- **Same calibration caveat as before**, plus a license caveat: the practice and test data are
  CC BY-NC-SA 4.0 (research use), so a pruned model refit on them should not be deployed
  commercially without the dataset authors' permission.

### d1 (2026-11-11): installing the skill without training, from a Jev-like teacher

**Problem.** The three experiments above put a frontier model in the loop: Opus reads activations, reasons,
and writes weights. The Kaggle notebook `fast-edits-decision-model` claims the same kind of result with no
LLM at all: learn the skill cheaply on the side with a small typed-decision model (the Jev / vegaml / Laya
family), then write its calibrated answers into a base model's weights with one closed-form solve, and show
that this beats LoRA on the same data while forgetting less. Question: does that hold on our insurance
benchmark, against our own rows, and how does it compare with Opus hand-editing BERT? Independent harness
in `wspd/` (above), base model Qwen3.5-0.8B, three split seeds (which 40 practice rows are kept back),
nothing calls an API ($0), all on the M1 laptop; runs under `runs/d1/`.

Three arms distil the same teacher on the same 1,456 rows and are scored by the same code on the 1,779
held-out sentences: the **direct edit** (one `down_proj`, 1,024 columns, ridge solve, dose line search,
guard cap), **LoRA** (rank 16, every MLP, 60 steps on the same loss), and the **unedited** reader as the
floor. The teacher itself is a row: it is a classifier too, just not one that generates text.

| arm | seeds | test acc (generated) | macro-F1 | test acc (scored) | agrees with teacher | ECE | selective acc @80% | weights changed | compute |
|---|---|---|---|---|---|---|---|---|---|
| unedited Qwen3.5-0.8B, short prompt | - | 58.07% | 0.5901 | 57.95% | 58.0% | 0.150 | 69.1% | 0 | - |
| unedited Qwen3.5-0.8B, full-description prompt (the 322-token prompt the edit does not use) | - | 61.21% | 0.6521 | 61.16% | 61.3% | 0.157 | 68.0% | 0 | - |
| **teacher: calibrated heads on the frozen reader (Jev-like)** | 3 | - | **0.9880** | **98.78%** (98.82 / 98.76 / 98.76) | - | 0.006 | 99.93% | 0 (head only) | ~10 min fit |
| **direct edit, layer 21 or 19, 1,024 columns, lambda 3** | 3 | **90.11%** (89.94 / 89.83 / 90.56) | 0.9020 | 90.03% (89.60 / 89.54 / 90.95) | 90.0% | 0.013 | 97.1% | **1.05M (0.14%)** | 50-76 ms solve; 21-26 min of forward passes; sweep 1 h 20 min |
| LoRA rank 16, 60 steps, same loss | 3 | 95.80% (96.40 / 96.29 / 94.72) | 0.9585 | 95.78% (96.40 / 96.29 / 94.66) | 95.5% | 0.015 | 99.6% | 5.3M trainable, 264M (35%) once merged | 66-80 min of fp32 training |
| Opus 5.5 edits BERT-base (v1/p50, 50 passes; different base model) | 1 | 98.31% | 0.9833 | | | | | 110M model rebuilt | $4.12, 50 passes |
| BERT fine-tune (3 ep) / linear probe / unedited | 1 | 97.25% / 88.53% / 13.10% | 0.9727 / 0.8870 / 0.0290 | | | | | | local GPU |

Seed values are in the order s0 / s1 / s2; means are over the seeds. Per-arm detail in `runs/d1/<arm>/eval/report.md`,
the full table in `runs/d1/compare.md`.

**Forgetting** (same 40 WikiText-2 paragraphs for every arm; ARC-Easy is 200 test questions drawn per seed,
so its "before" differs by seed; 40 capitals; 15 unrelated free generations checked for answer words and loops):

| arm | WikiText-2 ppl before -> after | KL to the unedited model (nats/token) | ARC-Easy before -> after | capitals | contaminated generations |
|---|---|---|---|---|---|
| direct edit s0 / s1 / s2 | 28.75 -> 29.98 / 29.99 / 31.26 | 0.032 / 0.033 / 0.067 | 68.5 -> 64.0 / 61.5 -> 57.5 / 66.0 -> 65.5 | 100% in all | 1 of 15 in all |
| LoRA s0 / s1 / s2 | 28.75 -> 30.98 / 29.67 / 30.03 | 0.076 / 0.081 / 0.050 | 68.5 -> 82.5 / 61.5 -> 44.5 / 66.0 -> 51.0 | 100% in all | 2 of 15 / 2 of 15 / 3 of 15 |

**Ablations** (seed 0, the chosen layer, 1,024 columns, each arm removed after measuring, scored at the answer
position; no generation pass):

| ablation | test acc (scored) | agrees with teacher | ECE | WikiText-2 ppl | ARC-Easy | contaminated | note |
|---|---|---|---|---|---|---|---|
| full method (re-run of seed 0) | 89.71% | 89.6% | 0.011 | 29.98 | 64.0% | 1 of 15 | reproduces direct_s0 (89.60%) |
| random columns, same solve | 88.87% | 88.6% | 0.019 | 30.46 | 63.5% | 0 of 15 | column choice buys under a point and a little less forgetting |
| no protection (lambda 0, no cap, no guard) | 57.95% | 58.0% | 0.150 | 28.75 | 68.5% | 0 of 15 | nothing written: every dose of the unregularized solve made the fit on the 40 kept-back rows worse than no edit, so the line search kept dose 0 (unedited scores) |
| hard gold labels (smoothed one-hot) instead of the teacher's probabilities | 94.77% | 94.8% | 0.165 | 29.25 | 66.5% | 1 of 15 | +5 points of accuracy, a second round accepted, calibration gone (ECE 0.011 -> 0.165) |
| 25% of the training rows (364) | 84.32% | 84.3% | 0.023 | 29.31 | 65.5% | 0 of 15 | rows matter: -5.4 points |
| 50% of the training rows (728) | 85.05% | 84.9% | 0.025 | 29.26 | 66.0% | 0 of 15 | -4.7 points |
| random change of the same size on the same columns | 54.69% | 54.6% | 0.197 | 31.58 | 63.0% | 1 of 15 | the solve, not the perturbation, carries the skill: -3 points vs unedited, +10% perplexity |

**Observations.**

- **The Jev-like teacher is the most accurate classifier in this repository**: 98.78% on the held-out set,
  above Opus's hand-edited BERT (98.31%) and the BERT fine-tune (97.25%), with an ECE of 0.006 and 99.93%
  accuracy on the 80% of sentences it is most confident about. It is a logistic head on two frozen features
  of Qwen3.5-0.8B (the final residual at the answer position and the mean layer-16 state over the sentence),
  fitted in about ten minutes of forward passes and a cross-validated sklearn fit. Same choice of features
  and C on all three seeds. Cheap, calibrated, abstains well; what it cannot do is generate.
- **The closed-form edit installs most of the skill, not all of it.** One round, 1.05M weights of one
  matrix (0.14% of the model), a 50 ms solve after about 25 minutes of forward passes: 58% -> 90%, with
  the first generated word being an option word on every test sentence. It then stops 8.7 points short
  of its teacher. The second round was refused on two seeds: every dose pushed the guard text past the
  0.03 nats/token cap, so the protection, not the solve, is the limit of a single-layer edit at this width.
  The kept-back rows said 85% to 95% agreement; the test set says 89.5% to 91.1%, which is within what 40
  rows can resolve.
- **LoRA on the same loss and rows wins on accuracy**: 94.7% to 96.4% (mean 95.8%), 5.7 points above the
  edit and 3 points below the teacher, at the cost of 66 to 80 minutes of fp32 gradient descent per seed on
  this laptop, touching every MLP matrix. This is the opposite of the notebook's headline on its 300-row
  tool-routing task. The difference is plausibly the task: eight classes with vocabulary-level boundaries
  that the unedited reader already gets 58% of, 1,456 teacher rows to fit, and two classes (conditions and
  duties, coverage grant) that the edit leaves at F1 0.76 and 0.84 while LoRA brings them to 0.94 and 0.97.
  A rank-16 update to 72 matrices has the capacity for that; 1,024 columns of one matrix under a guard cap
  do not.
- **Hard labels beat the calibrated teacher as edit targets here, at the price of calibration.** Writing
  smoothed one-hot gold labels instead of the teacher's probabilities scored 94.8% against 89.7%: the larger
  logit gaps let a second round through the guard cap. But the resulting model is badly over-confident (ECE
  0.165 against 0.011), which is the property the calibrated teacher was there to transfer. The notebook's
  finding that calibrated targets beat hard labels did not reproduce on accuracy; it did on calibration.
  Random columns cost under a point, so the column score is not where the method's value lies; halving the
  training rows costs about five points, so the data is.
- **Both arms forget a little, the edit less.** Edit: WikiText-2 perplexity +4% to +9%, ARC-Easy within
  noise on two seeds and -4.5 points on one, capitals intact, one of fifteen free generations picks up an
  answer word. LoRA: perplexity +3% to +8%, two or three contaminated generations, and ARC-Easy moves by +14, -17
  and -15 points on the same 200 questions across seeds, which says more about scoring a chat model's answer
  letter after 60 steps of a different prompt format than about knowledge. The no-protection ablation did not get far enough to damage anything: with lambda 0 the solve fits the
  1,456 training keys exactly and every dose raised the teacher gap on the 40 kept-back rows above the unedited
  value, so the dose search kept zero. The ridge term is what makes the edit generalise from the training rows
  at all, before it is what protects unrelated text. (In the smoke run on 64 rows the same ablation did go
  through, and took WikiText-2 perplexity from 27 to 644.)
- **Opus editing BERT remains the best weight-edit result on this benchmark (98.3%), but it is a
  different base model and a different mechanism**: a frontier model reasoning over activations and
  writing a 110M encoder's weights wholesale, against a formula writing 0.14% of a 752M decoder. The
  like-for-like comparison inside d1 is teacher vs edit vs LoRA on Qwen; the cross-experiment comparison
  is method plus base model and should be read as such.
- **The artifact is small and reproducible.** `edit.npz` is 4 MB; `check_rebuild.py` confirms base +
  bundle equals the baked 1.4 GB checkpoint tensor for tensor, and the hooked and baked models score the
  same test gap to four decimals (0.5224 vs 0.5224). The baked checkpoint loads with stock
  `AutoModelForCausalLM` and answers the test sentences with the option word as its first token.

### Checkpoints on the Hugging Face Hub

One public repo per run. The BERT and ClimateBERT checkpoints load with the stock `text-classification`
pipeline; see the calibration note above. The d1 repo is a causal LM (stock `AutoModelForCausalLM`) that answers
the classification prompt with one word, and it also carries the three edit bundles and the three teacher heads. The r1 checkpoints classify climate-disclosure paragraphs (ClimateBERT's
task, not the insurance task) and are research-use artifacts because their data is non-commercial. The pruned c1 checkpoint additionally met its accuracy bar by only 0.12
points on one seed and stores practice sentences as kernel anchors; its model card lists the caveats.

| checkpoint | setting | test acc |
|---|---|---|
| [wsp-qwen3.5-0.8b-insurance-d1-direct](https://huggingface.co/sidman007/wsp-qwen3.5-0.8b-insurance-d1-direct) | **d1: Qwen3.5-0.8B with 1,024 columns of one `down_proj` rewritten by a closed-form edit (0.14% of weights, no training)**, seed 0 baked; `edit/s{0,1,2}/` exact diffs, `teacher/s{0,1,2}/` Jev-like heads (98.8%) | 89.9% / F1 0.900 |
| [wsp-climatebert-tcfd-r1-p50](https://huggingface.co/sidman007/wsp-climatebert-tcfd-r1-p50) | **r1: pruned ClimateBERT TCFD classifier, 6 layers / FFN 1536, 68.1M params (-17.2%)**, 50 passes; research use (data CC BY-NC-SA) | 78.8% / F1 0.721 |
| [wsp-climatebert-tcfd-r1-p25](https://huggingface.co/sidman007/wsp-climatebert-tcfd-r1-p25) | r1: pruned ClimateBERT TCFD classifier, 6 layers / FFN 1792, 70.5M params (-14.3%), 25 passes; research use | 77.8% / F1 0.710 |
| [wsp-bert-insurance-c1-p50](https://huggingface.co/sidman007/wsp-bert-insurance-c1-p50) | **pruned: 8 layers / FFN 1024, 56.0M params (-48.9%)**, 50 passes | 97.9% |
| [wsp-bert-insurance-v1-p50](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p50) | v1, 50 passes (best) | 98.3% |
| [wsp-bert-insurance-v1-p25](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p25) | v1, 25 passes | 98.2% |
| [wsp-bert-insurance-v1-p10](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p10) | v1, 10 passes | 97.3% |
| [wsp-bert-insurance-v1-p00](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p00) | v1, 0 passes | 96.0% |
| [wsp-bert-insurance-v0-p25](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p25) | v0, 25 passes (self-written data) | 98.4% |
| [wsp-bert-insurance-v0-p00](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p00) | v0, 0 passes | 98.0% |
| [wsp-bert-insurance-v0-p50](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p50) | v0, 50 passes | 97.3% |
| [wsp-bert-insurance-v0-p10](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p10) | v0, 10 passes | 96.7% |
| [wsp-bert-insurance-v0-smoke](https://huggingface.co/sidman007/wsp-bert-insurance-v0-smoke) | plumbing test, Sonnet 5.5 | 87.9% |
| [wsp-bert-insurance-baseline-finetune](https://huggingface.co/sidman007/wsp-bert-insurance-baseline-finetune) | full fine-tune | 97.3% |
| [wsp-bert-insurance-baseline-probe](https://huggingface.co/sidman007/wsp-bert-insurance-baseline-probe) | linear probe | 88.5% |
| [wsp-bert-insurance-baseline-unedited](https://huggingface.co/sidman007/wsp-bert-insurance-baseline-unedited) | unmodified base, zero head | 13.1% |

## Run layout

```
runs/<version>/<name>/            # v0 = paper rules; v1 = + rule 6; c1 = pruning runs (wspc/); r1 = reinsurance pruning (wspr/)
  config.json  instructions.md  state.json  messages.json  driver.log
  turns/tNNN/{prompt.txt, answer.md, events.jsonl, run.json}
  requests/fwd_<k>.json          # each forward-pass request + result
  tool_calls.jsonl  sandbox_runs.jsonl  explanations.md  final_answer.md
  work/                          # the editor's persistent folder (edits/, pass_k/, notes.md, ...)
  edit/{edit.npz, manifest.json} # exact diff (c1: + structure.json, diff relative to runs/v1/p50)
  model/                         # runnable HF checkpoint (+ README.md model card), kept for HF push
                                 # (c1: pruned shape, + structure.json, param_count.json)
  eval/{scores.json, per_item.jsonl, report.md}
  audit.json
runs/d1/<arm>/                    # d1 = distillation (wspd/): teacher_s{0,1,2}, direct_s{0,1,2}, lora_s{0,1,2}, unedited, ablation/<name>
  log.txt  config.json  sweep.json  eval/{scores.json, report.md}
  edit/{edit.npz, manifest.json, rebuild_check.json}   # the column update, its hash, the base commit
  model/                          # baked fp16 checkpoint (seed 0 only; stock AutoModelForCausalLM); pushed with scripts/distill/push_hf.py
  teacher.json  targets.json      # teacher arms: feature spec + calibration, probabilities per practice row
```

## License

Apache 2.0 (code, logs, data and exported checkpoints). See `LICENSE`.

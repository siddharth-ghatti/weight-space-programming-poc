# Weight-space programming: executive summary of four experiments

**Date:** 2026-11-12 · **Status:** proof of concept complete, all code, logs and checkpoints published ·
**Total API spend across all experiments:** about $95

## The question

Can a model be adapted to a business task by *writing its weights directly*, with no gradient descent,
instead of fine-tuning? The "Sparks of Weight-Space Programming" study (rikiyat.github.io) showed a
frontier model doing this on research puzzles. We asked four progressively harder versions of that
question on insurance and reinsurance tasks:

1. **Build.** Can a frontier model turn a generic BERT into an insurance clause classifier by editing weights?
2. **Shrink.** Can the same editor then cut that classifier's size in half without losing accuracy?
3. **Shrink someone else's model.** Can it do that to a published third-party classifier it did not build?
4. **Remove the frontier model.** Can a small learned "decision model" plus a closed-form formula install a
   skill into a base model's weights, and how does that compare with LoRA?

Short answer: yes to 1, 2 and 3, with the editor matching or beating conventional methods at a cost of
single-digit dollars per model. For 4, the formula installs most of the skill (58% to 90%) but LoRA
beats it (96%), while the cheap decision model on its own is the best classifier in the repository (98.8%).

## The common method

Experiments 1 to 3 share one harness design, reimplemented independently per experiment so earlier
results stay reproducible (`wsp/`, `wspc/`, `wspr/`).

- **The editor** is Claude Opus 5.5 driven by a tool loop with three tools: run Python in a sandbox,
  request a forward pass, and submit. The sandbox has numpy, scipy and safetensors but no torch and no
  network, and it cannot run the model. The editor reads weight tensors, computes new values in closed
  form (ridge regression, LDA, naive Bayes, least squares, SVD) and saves them as named edits.
- **A forward-pass budget** (0, 10, 25 or 50) is the only way the editor can observe activations. Each
  pass returns hidden states, pooled vectors, logits or attention on practice sentences and costs one unit.
- **Rules, as in the paper:** no backward passes, optimizers or iterative fitting; no running the model
  inside the sandbox; embeddings frozen; closed-form math on weights and activations allowed. From the
  second round on, rule 6: no self-authored data, enforced by the harness.
- **Everything is an exact diff.** Each submission is a recorded set of tensor changes with hashes of the
  base and edited tensors, a plain-English explanation from the editor, and a stock Hugging Face checkpoint
  that rebuilds bit for bit from the public base weights plus the diff. A static lint rejects torch,
  autodiff and sklearn in the sandbox, and an audit script checks every run for rule violations.
- **Scoring** is on a held-out test set whose hash is checked before each evaluation. Conventional
  baselines (fine-tune, linear probe, truncate-and-fine-tune) are exported and scored by the same code.

Experiment 4 uses a fourth package (`wspd/`) with no frontier model in the loop; its method is
described in its own section.

## Experiment 1: build an insurance clause classifier (v0, v1)

**Task.** Label a sentence from an insurance document as one of eight clause types (coverage grant,
exclusion, premium and payment, deductibles and limits, claims procedure, definitions,
term/cancellation/renewal, conditions and duties). Data: 1,496 labelled practice sentences for the
editor, 1,779 held-out test sentences never shown to it. The data is synthetic, generated for the study
across seven lines of business and four writing registers. Base model: bert-base-uncased, 110M
parameters, with a zero-initialised classification head.

**What the editors built.** Each run invented its own mechanism:

- *0 passes:* never ran BERT. Tokenised the practice set, fit a naive-Bayes-plus-ridge keyword model
  and compiled it into layer 0 (keyword detectors), layer 1 (sentence averaging) and the head, turning
  layers 2 to 11 into no-ops. 96.0%.
- *10 passes:* rewired layer-8 attention so the [CLS] token receives the mean of all token states, made
  layers 9 to 11 pass-through, and fit a ridge plus shrinkage-LDA head on pooled features read back with
  a pass. 97.3%.
- *25 and 50 passes:* a kernel classifier written into the layer-11 feed-forward block: 1,496 GELU
  units centred on the practice sentences' layer-11 features, with a least-squares readout. 98.2% and
  98.3%. Only 9 of 50 passes were used.

| Method | Test accuracy | API cost |
|---|---|---|
| Unmodified BERT, zero head | 13.1% | - |
| Linear probe | 88.5% | - |
| Full fine-tune (industry standard) | 97.3% | local GPU |
| Weight editing, 0 passes | 96.0% | $1.34 |
| Weight editing, 10 passes | 97.3% | $3.43 |
| Weight editing, 25 passes | 98.2% | $8.64 |
| **Weight editing, 50 passes** | **98.3%** | **$4.12** |

Three of four editing runs matched or beat full fine-tuning. Runs took 16 to 90 editor turns and 8 to
55 minutes each. The best run changed 5.3% of the parameters and explained every change. A first
round (v0, $23.41) found a loophole: the strongest editor wrote 2,201 labelled sentences of its own to
improve its fit (98.4%). The second round (v1, $17.53) banned that, and accuracy held at 98.3%.

## Experiment 2: shrink the classifier (c1)

**Objective.** Start from the 98.3% model and minimise exported parameters subject to held-out
accuracy >= 97.8%. Only structural cuts that leave a stock checkpoint: drop whole encoder layers and
narrow every kept layer's feed-forward block to one width. Vocabulary and embeddings (23.8M) stay.
The editor repairs what it cuts with closed-form refits from forward-pass activations.

| Method | Parameters | Cut | Test accuracy | Meets bar | Cost |
|---|---|---|---|---|---|
| Start (v1, 50 passes) | 109.5M | - | 98.3% | yes | - |
| Weight editing, 10 passes | 95.3M | 13% | 97.4% | no | $7.59 |
| Weight editing, 25 passes | 95.3M | 13% | 98.3% | yes | $7.78 |
| **Weight editing, 50 passes** | **56.0M** | **49%** | **97.9%** | **yes** | **$19.39** |
| Truncate to 6 layers + fine-tune, 10 epochs | 67.0M | 39% | 98.2% | yes | local GPU |
| Truncate to 4 layers + fine-tune, 10 epochs | 52.8M | 52% | 97.6% | no | local GPU |
| Same shape as the editor (8 layers, FFN 1024) + fine-tune | 56.0M | 49% | 97.1% | no | local GPU |

With 10 or 25 passes the editor dropped two layers and stopped. With 50 passes it dropped four of
twelve layers, narrowed the feed-forward width of the remaining eight from 3,072 to 1,024, re-fitted
each one, and reached 49% of the whole model (63% of the encoder) at 97.9%. The conventional recipe at
that exact shape scores 97.1% and fails the bar. On a laptop CPU the compressed model is 2.2 to 2.5
times faster (15.8 ms vs 34.8 ms per sentence at batch 1) and the checkpoint shrinks from 438 MB to
224 MB. One caution: the editor's own cross-validated estimates ran 0.3 to 1 point optimistic, and the
10-pass run missed the bar because of it.

## Experiment 3: shrink a model we did not train, for reinsurance (r1)

**Use case.** Reinsurers must assess and report the climate exposure of cedents and large insureds, which
means sorting millions of paragraphs from annual and sustainability reports into the four TCFD pillars
(governance, strategy, risk management, metrics and targets), on CPU, ideally on premises. ClimateBERT's
TCFD classifier (University of Zurich; DistilRoBERTa, 6 layers, 82.3M parameters) is the public model
for this, released with its training data.

**What we did.** Same editor, same rules, same two levers, 25 or 50 passes, working only from the
model's own 1,000 public training paragraphs. Bar: within 1 accuracy point and 0.02 macro-F1 of the
published model on its 320 held-out test paragraphs (the F1 condition protects the 26-paragraph
governance class). We measured the published model at 77.5%; its authors report no figure.

| Method | Parameters | Cut (total / encoder) | Test accuracy | Macro-F1 | Meets bar | Cost |
|---|---|---|---|---|---|---|
| Published ClimateBERT TCFD | 82.3M | - | 77.5% | 0.710 | yes | - |
| Weight editing, 25 passes | 70.5M | 14% / 27% | 77.8% | 0.710 | yes | $9.56 |
| **Weight editing, 50 passes** | **68.1M** | **17% / 33%** | **78.8%** | **0.721** | **yes** | **$10.14** |
| Drop one layer, no retraining (best conventional that passes) | 75.2M | 9% / 16% | 78.8% | 0.702 | yes | - |
| Drop two layers, no retraining (best pair) | 68.1M | 17% / 33% | 77.2% | 0.685 | no | - |
| Truncate to 4 layers + fine-tune (best of 4 recipes) | 68.1M | 17% / 33% | 74.4% | 0.680 | no | local GPU |
| Same shape as the editor, narrow by magnitude + fine-tune | 68.1M | 17% / 33% | 75.6% | 0.679 | no | local GPU |

Both runs kept all six layers, cut the feed-forward width of each by 42% (25 passes) or 50%
(50 passes) using activation statistics, and re-solved the kept units in closed form to reproduce what
the dropped ones contributed. Retraining a truncated model on the 1,000 available paragraphs fails the
bar at every depth. The embedding table is 48% of this model, which caps the whole-model reduction; the
encoder itself lost a third. The 50-pass model classifies a paragraph in 17.3 ms against 25.2 ms. The
data is research-licensed (CC BY-NC-SA 4.0), so these checkpoints are research artifacts.

## Experiment 4: install a skill with no frontier model and no training (d1)

**Question.** A public Kaggle notebook (TypeSafe's "Jev" decision models) claims a closed-form formula
can write a small learned policy into a base model's weights and beat LoRA while forgetting less. We
rebuilt that recipe on our insurance benchmark with our own decision model and compared it with LoRA on
identical data. No Claude in the loop, $0 of API spend, three seeds, all on a laptop.

**Method.** Base model: Qwen3.5-0.8B, an open chat model that answers the classification question
correctly 58% of the time when simply asked.

1. *Teacher.* A calibrated multinomial logistic head on two features read from the frozen base model at
   the answer position (the final-norm input and the mean layer-16 residual over the sentence), fitted
   with cross-validation and one temperature. It outputs a probability over the eight answer words and
   can abstain. About ten minutes to fit.
2. *Direct edit.* One forward pass reads the inputs to layer 21's `down_proj` at the answer position
   and a covariance of everything to protect (other tokens, unrelated text). A readout Jacobian maps the
   wanted change in answer log-probabilities to a change in the residual stream. A MEMIT-style ridge
   solve, dW = R^T K (K^T K + λ n C)^-1, writes that change into 1,024 columns of one matrix. A dose
   line search on 40 kept-back rows picks how much to apply, refusing any dose that shifts unrelated
   text by more than 0.03 nats per token. The solve takes 50 to 76 ms and touches 0.14% of the weights.
   The result is baked into an fp16 checkpoint that loads with stock tooling.
3. *LoRA.* Rank 16 on every MLP matrix, 60 steps of Adam, same teacher targets, same sentences, same
   loss, best checkpoint on the same 40 kept-back rows, merged into the weights.

| Arm | Test accuracy (3 seeds) | What changed | Compute |
|---|---|---|---|
| Base model, unmodified | 58.1% | nothing | - |
| **Teacher: calibrated head on the frozen base model** | **98.8%** | nothing (a separate head) | 10 min |
| **Direct closed-form edit** | **90.1%** | 0.14% of weights, one matrix | 50 ms solve, ~25 min of forward passes |
| LoRA on the same teacher targets | 95.8% | 35% of weights once merged | 66 to 80 min per run |
| Reference: Claude edits BERT (experiment 1, different base model) | 98.3% | whole 110M model | $4 |
| Reference: BERT full fine-tune | 97.3% | whole 110M model | GPU |

Ablations at seed 0: random columns 88.9%, hard gold labels instead of teacher probabilities 94.8% (but
calibration destroyed, ECE 0.165 vs 0.013), half the training rows 85.1%, a random change of the same
size 54.7%. With protection switched off, the solve fit the practice rows exactly and every dose made
held-out rows worse, so the harness wrote nothing. Forgetting: perplexity on unrelated text rose 4% to
9% for both methods; a 200-question general-knowledge quiz moved a few points for the edit and swung
widely for LoRA.

**Reading.** The teacher is the best classifier in the repository and is calibrated (99.9% accurate on
the 80% of sentences it is most confident about). The formula installs most of the skill but stops nine
points short of its teacher, because the protection that keeps the rest of the model intact also limits
how far one matrix can move; that same protection is what makes the edit generalise at all. LoRA wins on
accuracy here, the opposite of the notebook's claim on its 300-row tool-routing task. The formula's real
advantages are milliseconds instead of an hour, a 4 MB diff instead of a new checkpoint, and a smaller
footprint on everything else the model knows. The comparison with experiment 1 is method plus base model,
not method alone.

## Results at a glance

| Experiment | Base model | Best result | Conventional comparison | API cost |
|---|---|---|---|---|
| 1. Build a classifier | BERT 110M | 98.3% by editing, 50 passes | fine-tune 97.3% | $17.53 (v1), $23.41 (v0) |
| 2. Shrink it | the 98.3% model | 49% fewer parameters at 97.9% | truncate + fine-tune at that shape 97.1%, fails | $34.76 |
| 3. Shrink a third-party model | ClimateBERT TCFD 82M | 17% fewer (33% of encoder) at 78.8%, up from 77.5% | every retraining recipe fails the bar | $19.70 |
| 4. No frontier model | Qwen3.5-0.8B | teacher 98.8%; formula 90.1%; LoRA 95.8% | LoRA | $0 |

## What we learned

- **The editors reuse the pretrained model rather than retrain it.** They located where the model
  already represents the information (sentence-level averages of mid-layer states, early-layer word
  cues), reserved a small channel for it, and wrote a closed-form classifier on top. Three of four
  build runs converged independently on the same idea: pool the sentence's mid-to-late-layer states
  and fit a closed-form head on them.
- **Observation budget matters, but most of it goes unused.** Runs with activation access beat the run
  without by 1 to 2 points, yet the build runs used 6 to 15 passes of 10 to 50. The pruning runs were
  different: the 50-pass pruner used 47 of 50 and that budget bought the 49% cut.
- **Closed-form protection is the hinge of the no-LLM method.** Without the ridge term against the
  covariance of unrelated activations, the direct edit overfits and the harness refuses to write. With
  it, a 50 ms solve moves a 752M-parameter model from 58% to 90% on the task.
- **A cheap head on a frozen model is hard to beat for classification.** If the business need is
  "classify sentences", the 98.8% calibrated teacher is the answer. Writing the skill into the weights
  is worth it when the model must *answer in text* without a side head, or when one base model must carry
  many small skills as diffs.
- **Everything is auditable.** Each result is a hashed diff on a public base model with an explanation,
  which matters for model-risk review in regulated settings and is something fine-tuning cannot offer.

## Limits and caveats

- **Synthetic data, upper-bound numbers.** The insurance data is separable largely by vocabulary; a
  classifier that never ran BERT scored 96%. Real policy documents will be harder. The 97% to 98% range
  should be read as an upper bound until re-run on a licensed real corpus with human labels.
- **One seed per setting** in experiments 1 to 3; differences of a point or less are within run-to-run
  variation. Experiment 4 ran three seeds.
- **Uncalibrated edited classifiers.** The BERT edits pick the right label but report flat probabilities
  (a scale artifact of the final layers); a one-scalar temperature fixes this but has not been applied.
  The d1 teacher is calibrated; the d1 direct edit inherits its calibration (ECE 0.013).
- **Small test sets in experiment 3** (320 paragraphs; one paragraph is 0.3 points) and in the d1
  forgetting suite (200 quiz questions, 15 free-text prompts).
- **Scale.** Edited models are 82M to 752M parameters. Whether the editor method holds on larger models
  or generative tasks is open; the original study only partly answers it.
- **Licences.** ClimateBERT's data is CC BY-NC-SA 4.0; the r1 checkpoints are research artifacts.
  Everything else is Apache-2.0.

## What production would require

Calibration on a held-out set with abstain thresholds; evaluation on real documents with a labelling
guide; robustness checks (other lines of business, OCR noise, long multi-clause sentences); several
seeds and a fixed acceptance threshold; model-risk documentation built from the edit manifests and
explanations the harness already produces; fairness and compliance review; serving (ONNX, quantisation,
drift monitoring). The lifecycle advantage is that a new version is a new diff and a new audit record
rather than a new fine-tune.

## Published artifacts

Code, data, logs and write-ups: https://github.com/siddharth-ghatti/weight-space-programming-poc

Checkpoints on the Hugging Face Hub, one repo per run (user `sidman007`). BERT and ClimateBERT
checkpoints load with the stock `text-classification` pipeline; the d1 repo is a causal LM that answers
the classification prompt with one word and also carries the edit bundles and teacher heads.

| Repo | What it is | Held-out result |
|---|---|---|
| `wsp-bert-insurance-v1-p50` | experiment 1 best: edited BERT, 50 passes | 98.3% |
| `wsp-bert-insurance-v1-p25`, `-p10`, `-p00` | experiment 1, other budgets | 98.2% / 97.3% / 96.0% |
| `wsp-bert-insurance-v0-*` | first round (self-authored data allowed) | 96.7% to 98.4% |
| `wsp-bert-insurance-baseline-finetune`, `-probe`, `-unedited` | conventional baselines | 97.3% / 88.5% / 13.1% |
| `wsp-bert-insurance-c1-p50` | experiment 2: 56.0M parameters, 49% smaller | 97.9% |
| `wsp-climatebert-tcfd-r1-p50`, `-p25` | experiment 3: pruned ClimateBERT, research use | 78.8% / 77.8% |
| `wsp-qwen3.5-0.8b-insurance-d1-direct` | experiment 4: baked direct edit of Qwen3.5-0.8B, plus `edit/s{0,1,2}/` diffs and `teacher/s{0,1,2}/` heads | 89.9% (teacher 98.8%) |

## Where to look in the repository

- `README.md`: full protocol, harness internals and per-run technical detail for every experiment.
- `docs/executive-summary.md`: the longer business-facing summary with production checklist.
- `docs/exec-memo-model-pruning.md`: feasibility memo on the pruning method alone (experiments 2 and 3).
- `runs/compare.md`, `runs/c1/compare.md`, `runs/r1/compare.md`, `runs/d1/compare.md`: comparison tables.
- `runs/<experiment>/<run>/`: per-run logs, edits with hashes, editor explanations, scores and audits.
- Harnesses: `wsp/` (build), `wspc/` (shrink), `wspr/` (reinsurance), `wspd/` (distillation).

# Executive summary: building a classifier by editing model weights directly

**Date:** 2026-11-04 · **Status:** proof of concept complete, including a compression follow-up · **Recommendation:** continue investment

## What we did

We tested a new way to adapt a language model to a business task. Instead of fine-tuning (the
standard approach: run thousands of gradient-descent steps on labeled examples), we gave a frontier
model (Claude Opus 5.5) direct read and write access to the weights of a small open model
(BERT, 110M parameters) and asked it to turn that model into an insurance sentence classifier
by reasoning about the weights and writing new values in closed form. No gradient descent, no
training loop, and only a small, fixed budget of forward passes (0, 10, 25 or 50) to observe the
model's internal activations.

The task: label a sentence from an insurance document as one of eight types (coverage grant,
exclusion, premium and payment, deductibles and limits, claims procedure, definitions,
term/cancellation/renewal, conditions and duties). We used 1,496 labeled sentences for the editor to
work from and 1,779 held-out sentences, never shown to the editor, for scoring. The data is synthetic,
generated for this study across seven lines of business and four writing registers.

The approach replicates the protocol of the "Sparks of Weight-Space Programming" study
(rikiyat.github.io), applied to a business classification task instead of research puzzles.

## Results

Accuracy on the held-out test set. Conventional methods on the same labeled data are shown for comparison.

| Method | Test accuracy | Cost (API) |
|---|---|---|
| Unmodified model (no adaptation) | 13.1% | - |
| Linear probe (classical lightweight adaptation) | 88.5% | - |
| **Full fine-tuning (industry standard)** | **97.3%** | - (local GPU) |
| Weight editing, 0 forward passes | 96.0% | $1.34 |
| Weight editing, 10 forward passes | 97.3% | $3.43 |
| Weight editing, 25 forward passes | 98.2% | $8.64 |
| **Weight editing, 50 forward passes** | **98.3%** | **$4.12** |

Key points:

- **It works.** Three of four weight-editing runs matched or beat full fine-tuning on the held-out
  test set. The best run reached 98.3% accuracy, one point above fine-tuning.
- **It is cheap.** The entire four-run experiment cost $17.53 in API calls. The most expensive single
  run cost $8.64. An earlier, less restricted round of four runs cost $23.41.
- **It is fast.** Runs finished in 16 to 90 editor turns, between 8 and 55 minutes of wall-clock
  time each, all four running in parallel on one laptop.
- **It is auditable.** Every change is a recorded diff to named tensors with a plain-English
  explanation from the editor. In the best run, only 5.3% of the model's parameters changed, and the
  editor described exactly what each change does. Fine-tuning changes every parameter with no
  explanation.
- **Output is standard.** Each result is a normal Hugging Face checkpoint that loads with stock
  tooling and can be published or deployed with no custom code.

## Follow-up: shrinking the model by the same method

After the classifier was built, we asked whether the same approach could make it cheaper to run.
Starting from the best model above (98.3%, 109M parameters), the editor was told to remove as many
parameters as possible while keeping held-out accuracy at or above 97.8%, using only changes that
keep the result a standard model: dropping whole layers and narrowing each layer's feed-forward
block. The vocabulary and embeddings (23.8M parameters) stay intact so the model keeps generalizing
to unseen wording. The editor still could not train anything; it had to decide what to cut and
repair the rest with closed-form fits from a budget of forward passes.

| Method | Size | Reduction | Test accuracy | Meets 97.8% bar | Cost |
|---|---|---|---|---|---|
| Starting model (v1, 50 passes) | 109.5M | - | 98.3% | yes | - |
| Weight editing, 10 forward passes | 95.3M | 13% | 97.4% | no | $7.59 |
| Weight editing, 25 forward passes | 95.3M | 13% | 98.3% | yes | $7.78 |
| **Weight editing, 50 forward passes** | **56.0M** | **49%** | **97.9%** | **yes** | **$19.39** |
| Conventional: keep 6 layers, fine-tune (10 epochs) | 67.0M | 39% | 98.2% | yes | local GPU |
| Conventional: keep 4 layers, fine-tune (10 epochs) | 52.8M | 52% | 97.6% | no | local GPU |
| Conventional at the editor's exact size (8 layers, narrow FFN), fine-tune | 56.0M | 49% | 97.1% | no | local GPU |

Key points:

- **Half the model, same accuracy bar.** The 50-pass run removed 49% of the parameters and kept
  97.9% accuracy. The conventional recipe (truncate the model, then fine-tune) fails the bar at that
  size: 97.1% at the identical shape, and it needs 67M parameters (a 39% cut) to pass.
- **It is a faster model, not just a smaller file.** On a laptop CPU the compressed model classifies a sentence in 15.8 ms
  one at a time and 5.8 ms in batches, against 34.8 ms and 14.8 ms for the original: 2.2 to 2.5 times
  faster. The checkpoint shrinks
  from 438 MB to 224 MB.
- **Budget bought the result.** With 10 or 25 passes the editor took a safe 13% cut and stopped. With
  50 passes it rebuilt every remaining layer one at a time and reached 49%. The whole three-run
  experiment cost $34.76.
- **One run missed the bar.** The 10-pass run estimated its own accuracy at 98.5% from
  cross-validation on the practice data and scored 97.4% on the held-out set. The editor's internal
  estimates run 0.3 to 1 point optimistic; the margin to the bar has to be set with that in mind.
- **Still auditable and still standard.** The compressed model is a normal checkpoint that loads
  with stock tooling, and it can be rebuilt exactly from the public base model plus two recorded
  diffs. The editor's explanation of what it kept, what it cut and how it repaired each layer is in
  the run log.

For deployment this changes the picture: a 56M-parameter encoder serves at a fraction of the cost of
the 110M original on CPU, with no GPU, which puts per-sentence classification of whole policy
documents within reach of commodity infrastructure. The production caveats below still apply.

## What we learned

- **The editor uses the pretrained model intelligently.** Rather than retraining, the editors
  located where the model already represents the information they needed (sentence-level
  averages of mid-layer states, early-layer word cues), reserved a small channel in the network to
  carry it, and wrote a classifier on top. Three of four runs invented the same mechanism
  independently.
- **Observation budget matters more than it costs.** Runs with activation access beat the run
  without it by 1 to 2 points, yet most of the forward-pass budget went unused (9 to 15 passes in the
  runs with 10 to 50 available).
- **A first round exposed a loophole we then closed.** In the first round, the strongest run
  authored 2,201 extra labeled sentences of its own to improve its fit. That is a legitimate
  capability but it muddies the comparison, so the second round banned self-authored data and
  enforced the ban in the harness. Accuracy held: 98.3% without the shortcut versus 98.4% with it.

## Why this is worth pushing on

- **Cost and speed:** single-digit dollars and under an hour per adapted model, with no GPU
  training infrastructure.
- **Transparency:** a readable record of what changed and why, which matters for regulated use
  such as insurance and for model-risk review.
- **Small footprint:** edits touch 5 to 10% of parameters and are shipped as an exact diff, so a
  base model can carry many task adaptations.
- **Headroom:** the editors under-used their budgets and worked from a small synthetic dataset.
  Harder, real-world data and more observation are the obvious next experiments.
- **It compresses too:** the same method, pointed at size instead of accuracy, halved the model while
  holding the accuracy bar, where the standard compression recipe at that size fell short.

## Caveats

- **Accurate, but uncalibrated.** The edited models pick the right label, but the probability scores
  they report are flat (about 0.2 for the winning label instead of 0.9+). This is a scale artifact of
  how the editors wired the final layers, not uncertainty: rescaling the logits by a constant leaves
  every prediction unchanged and restores normal-looking confidences. It also means the scores
  cannot be used as confidence estimates until a calibration step is added (see follow-ups). The
  fine-tuned baseline is not calibrated either, but its decision margins are larger in absolute
  terms (median gap between the top two logits 3.1 versus 0.3), so a small fraction of edited-model
  predictions (about 3% of test sentences, against 0.4% for fine-tuning) sit close to a flip.
- One run per setting; differences of a point or less are within likely run-to-run variation.
- The data is synthetic and largely separable by vocabulary. Real insurance documents will be harder.
  The fine-tuning baseline would also face that difficulty, so the comparison should be re-run on
  real data before any production decision.
- This is a 110M-parameter encoder. Whether the approach scales to larger models or generative tasks
  is an open question the original study only partly answers.

## Recommended next steps (research)

1. Re-run on a real, licensed insurance corpus with human labels.
2. Run three or more seeds per setting to measure variance.
3. Try a second base model and a multi-label or span-level task.
4. Publish the code, logs and checkpoints so others can reproduce the result (done; see below).

## What it would take to put one of these models in production

None of the checkpoints is production-ready as it stands. The gap is the same one any new
classifier would have to close, plus two items specific to this method.

1. **Calibration.** Fit a temperature (one scalar) on a held-out set so the reported probabilities
   mean what they say, then set abstain and human-review thresholds from the calibrated scores.
   Mechanically this is a one-line rescale of the classifier weights; the work is in the held-out
   evaluation.
2. **Real data and a real test set.** Evaluate on sentences from actual policy documents, claims
   correspondence and customer support, with human labels and a documented labeling guide. The
   synthetic data here is separable mostly by vocabulary, so today's 98% is an upper bound on what
   to expect.
3. **Robustness checks.** Out-of-distribution lines of business, long and multi-clause sentences,
   OCR noise, non-English or mixed-register text, and adversarial phrasing. Report per-class
   precision and recall, not just accuracy, because exclusion and coverage-grant errors have
   asymmetric cost.
4. **Variance and acceptance criteria.** Several seeds per configuration, a fixed acceptance
   threshold agreed with the business owner, and a regression test suite that every new edit must
   pass before release.
5. **Model-risk and audit documentation.** Specific to this method: a signed record of the exact
   base-model hash, the edit diff and its hashes (already produced by the harness as
   `edit_manifest.json`), the editor's explanation, and the audit report showing no training or
   external data was used. Review the edit explanation with a domain expert; it is readable in a way
   fine-tuned weights are not.
6. **Fairness and compliance review.** Check behaviour across lines of business and customer
   segments, and confirm the classifier's use fits the applicable insurance-regulation and
   model-governance requirements for the jurisdictions involved.
7. **Serving.** Export to ONNX or TorchScript, quantize if latency matters, load-test, and add
   monitoring for prediction-distribution drift and calibration drift over time.
8. **Lifecycle.** Decide how retraining works: because an adapted model is a small diff on a
   public base model, a new version is a new diff and a new audit record, which is cheaper and
   more traceable than a new fine-tune.

## Published artifacts

Code, logs, data and this write-up: https://github.com/siddharth-ghatti/weight-space-programming-poc

Checkpoints on the Hugging Face Hub (standard `BertForSequenceClassification`, load with the stock
`text-classification` pipeline). All carry the calibration caveat above. The compressed checkpoint
cleared its 97.8% bar by 0.12 points on a single seed and should be re-validated before any use.

| checkpoint | setting | held-out accuracy |
|---|---|---|
| [wsp-bert-insurance-c1-p50](https://huggingface.co/sidman007/wsp-bert-insurance-c1-p50) | **compressed: 56.0M parameters (49% smaller)**, 50 passes | 97.9% |
| [wsp-bert-insurance-v1-p50](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p50) | v1, 50 passes (best) | 98.3% |
| [wsp-bert-insurance-v1-p25](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p25) | v1, 25 passes | 98.2% |
| [wsp-bert-insurance-v1-p10](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p10) | v1, 10 passes | 97.3% |
| [wsp-bert-insurance-v1-p00](https://huggingface.co/sidman007/wsp-bert-insurance-v1-p00) | v1, 0 passes | 96.0% |
| [wsp-bert-insurance-v0-p25](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p25) | v0, 25 passes (used self-written data) | 98.4% |
| [wsp-bert-insurance-v0-p00](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p00) | v0, 0 passes | 98.0% |
| [wsp-bert-insurance-v0-p50](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p50) | v0, 50 passes | 97.3% |
| [wsp-bert-insurance-v0-p10](https://huggingface.co/sidman007/wsp-bert-insurance-v0-p10) | v0, 10 passes | 96.7% |
| [wsp-bert-insurance-v0-smoke](https://huggingface.co/sidman007/wsp-bert-insurance-v0-smoke) | plumbing test, Sonnet 5.5 | 87.9% |
| [wsp-bert-insurance-baseline-finetune](https://huggingface.co/sidman007/wsp-bert-insurance-baseline-finetune) | full fine-tune baseline | 97.3% |
| [wsp-bert-insurance-baseline-probe](https://huggingface.co/sidman007/wsp-bert-insurance-baseline-probe) | linear-probe baseline | 88.5% |
| [wsp-bert-insurance-baseline-unedited](https://huggingface.co/sidman007/wsp-bert-insurance-baseline-unedited) | unmodified base, zero head | 13.1% |

## Where things are in the repository

- Code and harness: `README.md` has the full protocol and per-run technical detail.
- Per-run logs, edits, explanations and scores: `runs/v0/` (first round) and `runs/v1/` (second round).
- Comparison table across all runs: `runs/compare.md`.
- Pruning experiment: harness in `wspc/`, runs and logs in `runs/c1/`, conventional compression
  baselines in `runs/_baselines/compress/`, comparison table `runs/c1/compare.md`.
- Reinsurance experiment: harness in `wspr/`, runs in `runs/r1/`, baselines in `runs/_baselines/reins/`,
  comparison table `runs/r1/compare.md`, data in `data/tcfd/`.

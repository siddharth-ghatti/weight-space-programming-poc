# Executive summary: building a classifier by editing model weights directly

**Date:** 2026-11-04 · **Status:** proof of concept complete · **Recommendation:** continue investment

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

## Caveats

- One run per setting; differences of a point or less are within likely run-to-run variation.
- The data is synthetic and largely separable by vocabulary. Real insurance documents will be harder.
  The fine-tuning baseline would also face that difficulty, so the comparison should be re-run on
  real data before any production decision.
- This is a 110M-parameter encoder. Whether the approach scales to larger models or generative tasks
  is an open question the original study only partly answers.

## Recommended next steps

1. Re-run on a real, licensed insurance corpus with human labels.
2. Run three or more seeds per setting to measure variance.
3. Try a second base model and a multi-label or span-level task.
4. Publish the code, logs and checkpoints so others can reproduce the result.

## Where things are

- Code and harness: this repository (`README.md` has the full protocol and per-run technical detail).
- Per-run logs, edits, explanations and scores: `runs/v0/` (first round) and `runs/v1/` (second round).
- Comparison table across all runs: `runs/compare.md`.

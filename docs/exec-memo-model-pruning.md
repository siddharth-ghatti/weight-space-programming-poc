# Executive memo: shrinking classification models by weight editing

**Subject:** Feasibility of LLM-driven weight editing as a model compression method
**Date:** 2026-11-06
**Status:** two proof-of-concept experiments complete; recommendation is to proceed to a pilot on licensed data
**Supporting detail:** `docs/executive-summary.md` (all three experiments), `README.md` (full protocol and per-run results)

## The question

Our document-classification models run on CPU, over millions of sentences or paragraphs, every
renewal cycle. Smaller models mean lower serving cost, lower latency, and the option to run
on-premises. The standard way to shrink a model is to cut layers and retrain it, which needs a
training pipeline, a GPU, enough labeled data to retrain on, and produces a result nobody can
explain beyond "we retrained it".

We tested a different method: give a frontier model (Claude Opus 5.5) direct read and write access
to the weights of a small classification model inside a sandbox, state a size goal and an accuracy
bar, and let it decide what to remove and how to repair what is left. It cannot train anything. It
can only inspect weights, observe the model's internal activations on a fixed budget of forward
passes (25 or 50), and write new weight values it computes in closed form. Everything it does is
logged, and the output is a standard checkpoint that loads with stock tooling.

The method originates in the "Sparks of Weight-Space Programming" study. We ran it on two
compression problems that matter to us.

## What we ran

**Experiment 1: compress a classifier we built.** Start model: a 110M-parameter BERT insurance
sentence classifier at 98.3% accuracy on 1,779 held-out sentences. Goal: as small as possible while
staying at or above 97.8%. Allowed changes: drop whole layers, narrow the feed-forward blocks.
Embeddings (the vocabulary table) stay intact.

**Experiment 2: compress a classifier somebody else built, for a reinsurance use case.** Start
model: ClimateBERT's published TCFD disclosure classifier (82M parameters, University of Zurich),
which sorts passages of annual and sustainability reports into the four TCFD pillars. This is the
public model for screening cedents' and corporate insureds' climate-risk disclosures at portfolio
scale. Goal: as small as possible while staying within 1 accuracy point and 0.02 macro-F1 of the
published model on its own 320 held-out test paragraphs. The editor had only the model's public
training paragraphs to work from (1,000), which is the position a reinsurer is actually in: it did
not train the model and cannot retrain it from scratch.

In both cases we ran the conventional alternatives on the same data for comparison: truncate and
retrain at several depths, retrain at the editor's exact final shape, and drop layers with no
retraining.

## Results

| | Experiment 1: our classifier | Experiment 2: ClimateBERT TCFD |
|---|---|---|
| Start model | 109.5M params, 98.3% | 82.3M params, 77.5% / F1 0.710 |
| **Weight-edited model (50 passes)** | **56.0M params (49% smaller), 97.9%** | **68.1M params (17% smaller, 33% of the encoder), 78.8% / F1 0.721** |
| Meets the bar | yes | yes |
| Best conventional at the same size | 97.1%, fails | 75.6% / F1 0.679, fails |
| Best conventional that passes | 67.0M (39% smaller), retrained 10 epochs | 75.2M (9% smaller), one layer dropped, no retraining |
| CPU latency vs start (single item) | 15.8 ms vs 34.8 ms, 2.2x faster | 17.3 ms vs 25.2 ms, 1.5x faster |
| Checkpoint size | 224 MB vs 438 MB | 273 MB vs 329 MB |
| API cost, all runs in the experiment | $34.76 (three runs) | $19.70 (two runs) |
| Wall time per run, laptop, no GPU | under 2 h 30 min | under 1 h 40 min |

Three findings carry the weight:

1. **The edited models beat retraining at matched size, in both experiments.** At the exact shape
   the editor chose, the conventional truncate-and-retrain recipe missed the bar by 0.8 points in
   experiment 1 and by 2 to 3 points in experiment 2. In experiment 2, retraining failed at every
   depth we tried, from either starting point, because 1,000 paragraphs is not enough data to
   retrain a truncated model. The editor did not need to retrain. In experiment 2 it kept all six
   layers, cut the feed-forward width of each by 42% (25 passes) or 50% (50 passes) using
   activation statistics, which removed 28% and 33% of the encoder's parameters, and re-solved the
   remaining weights so they reproduce what the removed units contributed.
2. **The compressed model is not worse at the task.** Experiment 2 is the cleaner test because the
   test set is the published model's own: the edited model scored slightly higher than the original
   (within noise on 320 paragraphs, which is the point). In experiment 1, accuracy dropped 0.4
   points for a 49% reduction.
3. **Budget buys result.** With 10 or 25 forward passes the editor dropped 2 of 12 layers (a 13%
   cut of the whole model, 17% of the encoder) and stopped. With 50 passes it dropped 4 of 12 layers,
   narrowed the feed-forward width of the remaining eight from 3072 to 1024 (a two-thirds cut), and
   re-fitted them, reaching 49% of the whole model and 63% of the encoder. The 50-pass runs cost
   $10 to $19 each.

## Why this is usable, not just interesting

- **Cost.** Single-digit to low-double-digit dollars of API spend per compressed model. No GPU, no
  training pipeline, no MLOps investment to get started. The experiments ran on a laptop.
- **Auditability.** This is the property that matters most for model-risk review. The output is a
  diff against a pinned base-model hash, every tensor change is listed with its hash, the editor
  writes a plain-English account of what it removed and why, and the harness produces an audit
  report confirming no training and no outside data were used. A fine-tuned model has none of this.
- **Works without training data.** Compression by retraining needs enough labeled data to recover
  from the cut. This method needs only enough data to observe the model's behaviour, which is a
  much lower bar, and it is the bar we actually face with vendor and open-source models.
- **Standard artifact.** The result loads with stock Hugging Face tooling, exports to ONNX or
  TorchScript like any other checkpoint, and is rebuildable bit-for-bit from the public base model
  plus the recorded diff.

## What it is not yet

- **One run per setting.** Differences of a point or less are within likely run-to-run variation.
  A production decision needs three or more seeds per configuration.
- **Small test sets.** 1,779 synthetic sentences in experiment 1, 320 real paragraphs in experiment
  2. Experiment 1's data is synthetic and separable mostly by vocabulary, so 98% is an upper bound.
- **The editor's self-estimates run optimistic** by 0.3 to 1 point. One of the five pruning runs
  (experiment 1, 10 passes) missed its bar for exactly this reason. Acceptance must be decided by a
  held-out test the editor never sees, which is how we ran it, and the margin to the bar must be set
  with that optimism in mind.
- **Calibration.** The models pick the right label but their probability scores are not usable as
  confidences without a one-parameter calibration step on held-out data.
- **License.** The ClimateBERT training data is research-use only (CC BY-NC-SA 4.0). The experiment 2
  model is a research artifact. A commercial deployment would need the dataset authors' permission or
  a refit on licensed data.
- **Scale is untested.** Both start models are small encoders (80M to 110M parameters). Whether the
  method holds for larger models or generative tasks is an open question.

## Recommendation

Proceed to a pilot with a defined scope: one production classification model, a licensed held-out
test set with human labels, three seeds per configuration, calibration, and a model-risk review of
the edit record. The expected cost of the compute side is under $200 in API spend; the cost is in
the evaluation data and the review, which any compression method would also require.

The upside, if the pilot holds: compressed models at roughly half the serving cost, produced in an
afternoon for tens of dollars, with an audit trail we can hand to a reviewer. The method also
applies to adapting a base model to a new task, which the first experiment in this repository
demonstrated, so a single harness covers both compression and adaptation.

## Published artifacts

All code, run logs, editor explanations and checkpoints are public.

- Repository: `siddharth-ghatti/weight-space-programming-poc` on GitHub (Apache-2.0).
- Experiment 1 checkpoint: [sidman007/wsp-bert-insurance-c1-p50](https://huggingface.co/sidman007/wsp-bert-insurance-c1-p50)
- Experiment 2 checkpoints: [sidman007/wsp-climatebert-tcfd-r1-p50](https://huggingface.co/sidman007/wsp-climatebert-tcfd-r1-p50),
  [sidman007/wsp-climatebert-tcfd-r1-p25](https://huggingface.co/sidman007/wsp-climatebert-tcfd-r1-p25) (research use)
- Per-run comparison tables: `runs/c1/compare.md`, `runs/r1/compare.md`.

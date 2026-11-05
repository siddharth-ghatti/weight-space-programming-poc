# Evaluation: `runs/r1/p25/model`

**Test accuracy 0.7781 | macro-F1 0.7104 | weighted-F1 0.7788 | n=320**
Practice accuracy 0.9100 | macro-F1 0.9006 | gap +0.1319 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.633 | 0.731 | 0.679 | 26 |
| risk | 0.643 | 0.562 | 0.600 | 48 |
| strategy | 0.868 | 0.838 | 0.853 | 197 |
| metrics | 0.655 | 0.776 | 0.710 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 19 | 2 | 4 | 1 |
| 1 risk | 5 | 27 | 14 | 2 |
| 2 strategy | 5 | 10 | 165 | 17 |
| 3 metrics | 1 | 3 | 7 | 38 |

## Edited tensors

12 tensors, 14,160,382 entries changed (17.2054% of the start model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| roberta.encoder.layer.0.output.dense.bias | 768 | 1.0000 | 2.247 | 2.279 |
| roberta.encoder.layer.0.output.dense.weight | 2,359,296 | 1.0000 | 47.109 | 73.306 |
| roberta.encoder.layer.1.output.dense.bias | 768 | 1.0000 | 2.039 | 2.508 |
| roberta.encoder.layer.1.output.dense.weight | 2,359,296 | 1.0000 | 50.685 | 77.958 |
| roberta.encoder.layer.2.output.dense.bias | 768 | 1.0000 | 2.055 | 2.173 |
| roberta.encoder.layer.2.output.dense.weight | 2,359,296 | 1.0000 | 47.255 | 71.199 |
| roberta.encoder.layer.3.output.dense.bias | 768 | 1.0000 | 2.989 | 1.730 |
| roberta.encoder.layer.3.output.dense.weight | 2,359,296 | 1.0000 | 49.892 | 72.692 |
| roberta.encoder.layer.4.output.dense.bias | 768 | 1.0000 | 2.920 | 1.563 |
| roberta.encoder.layer.4.output.dense.weight | 2,359,295 | 1.0000 | 47.305 | 70.436 |
| roberta.encoder.layer.5.output.dense.bias | 768 | 1.0000 | 14.971 | 1.637 |
| roberta.encoder.layer.5.output.dense.weight | 2,359,295 | 1.0000 | 37.826 | 57.806 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 70,497,796 | 31,316,740 | 6 | 1792 | 282.0 MB | 14.3% | 27.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 18.678 ms/paragraph, batch 16: 20.798 ms/paragraph

Target: **PASS** — accuracy >= 76.50% (77.81%: ok); macro-F1 >= 0.6896 (0.7104: ok)

## Editor's final answer

I kept all 6 encoder layers and narrowed every feed-forward block from 3072 to 1792 units. The model goes from 82,301,956 to 70,497,796 exported parameters: 14.3% smaller overall, and 27.4% smaller outside the frozen embeddings. Whether it clears the test bar is uncertain. My rough estimate is about a 1-in-5 chance it misses (more than 4 of roughly 400 test items lost), and the practice numbers alone don't settle it.

**Practice results (all 1000 paragraphs, checked by a real forward pass):**
- It agrees with the original model's prediction on 98.1% of paragraphs (19 differ).
- Accuracy is 0.910 against 0.925 for the original, and macro-F1 is 0.9006 against 0.9136.
- Recall is unchanged for governance and risk; strategy and metrics each lose about 2 points.
- The 1.5-point accuracy drop is bigger than the 1.0-point test allowance. I think practice overstates the test damage, because the practice set is the model's own training data. The pruned model loses some of what it memorised: on low-confidence paragraphs it drifts away from the training label whether the original was right or wrong. On unseen test paragraphs, prediction changes should behave more like random noise, but this is a judgement, not something I could measure.

**How each layer was narrowed (closed-form only, no training):**
1. I read the actual inputs to each feed-forward block from the original model on practice paragraphs. That was token-level for layers 0–4 (about 14,000 tokens for layers 0–2, from 32 random plus 32 of the longest paragraphs, and about 4,000 for layers 3–4). For layer 5 I used only the `<s>` position of all 1000 paragraphs, since that is the only position the classifier reads.
2. In each layer I kept the units with the largest output-weight energy times activation variance.
3. I re-solved the kept units' output weights and the output bias by ridge-regularised least squares. This makes the kept units reproduce the full block's output, including the contribution of the dropped units that they can predict. Input weights, attention, LayerNorms and the classifier head are unchanged.

**Why this size and setup:**
- **Dropping a whole layer** was much worse in every variant I tried (agreement 0.94–0.955 even with the head refitted).
- **m=1536** (68.1M params) reached only 97.7% agreement, its practice macro-F1 drop (0.021) already exceeds the 0.02 bar, and its prediction changes on borderline paragraphs were about 28% larger.
- **m=2048** (72.9M) had 15 changed predictions against 19, a small gain in safety for 2.4M more parameters.
- **A layer-5 fit trained on the pruned model's own inputs** looked better on practice but did worse under item-level cross-validation (it overfit the 1000 practice items). I replaced it with the plain fit, which matches an unpruned layer 5.
- **Tried with no real gain:** more probe data, downstream-weighted unit selection, a free diagonal correction absorbed into a LayerNorm, and a ridge correction of the classifier head.

The structure is in `structure.json` (m=1792; the kept unit ids per layer are also in `keep_F1792.json`). Four forward passes were left unused.

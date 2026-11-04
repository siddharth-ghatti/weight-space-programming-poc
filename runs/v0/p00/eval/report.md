# Evaluation: `<repo>/runs/p00/model`

**Test accuracy 0.9798 | macro-F1 0.9799 | weighted-F1 0.9798 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0202

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.978 | 0.974 | 0.976 | 233 |
| exclusion | 0.987 | 0.973 | 0.980 | 226 |
| premium_payment | 0.944 | 0.995 | 0.969 | 219 |
| deductible_limits | 1.000 | 0.978 | 0.989 | 225 |
| claims_procedure | 0.991 | 0.986 | 0.989 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 0.973 | 0.955 | 0.964 | 223 |
| conditions_duties | 0.978 | 0.978 | 0.978 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 227 | 2 | 1 | 0 | 0 | 1 | 1 | 1 |
| 1 exclusion | 2 | 220 | 0 | 0 | 0 | 1 | 1 | 2 |
| 2 premium_payment | 0 | 0 | 218 | 0 | 0 | 0 | 1 | 0 |
| 3 deductible_limits | 2 | 0 | 2 | 220 | 0 | 0 | 1 | 0 |
| 4 claims_procedure | 0 | 1 | 1 | 0 | 219 | 0 | 0 | 1 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 0 | 9 | 0 | 0 | 0 | 213 | 1 |
| 7 conditions_duties | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 220 |

## Edited tensors

108 tensors, 40,169,480 entries changed (36.6883% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.0.attention.output.LayerNorm.bias | 768 | 1.0000 | 9.131 | 9.131 |
| bert.encoder.layer.0.attention.output.LayerNorm.weight | 768 | 1.0000 | 3.794 | 26.663 |
| bert.encoder.layer.0.attention.output.dense.bias | 768 | 1.0000 | 0.983 | 0.983 |
| bert.encoder.layer.0.attention.output.dense.weight | 589,824 | 1.0000 | 21.947 | 21.947 |
| bert.encoder.layer.0.intermediate.dense.bias | 3,072 | 1.0000 | 325.952 | 6.588 |
| bert.encoder.layer.0.intermediate.dense.weight | 2,359,296 | 1.0000 | 62.630 | 57.479 |
| bert.encoder.layer.0.output.LayerNorm.bias | 768 | 1.0000 | 2.912 | 2.912 |
| bert.encoder.layer.0.output.LayerNorm.weight | 768 | 1.0000 | 6.943 | 21.005 |
| bert.encoder.layer.0.output.dense.bias | 768 | 1.0000 | 2.222 | 2.222 |
| bert.encoder.layer.0.output.dense.weight | 2,359,296 | 1.0000 | 56.325 | 55.062 |
| bert.encoder.layer.1.attention.output.LayerNorm.bias | 768 | 1.0000 | 5.384 | 5.384 |
| bert.encoder.layer.1.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.217 | 24.390 |
| bert.encoder.layer.1.attention.output.dense.bias | 768 | 1.0000 | 1.673 | 1.673 |
| bert.encoder.layer.1.attention.output.dense.weight | 589,824 | 1.0000 | 24.295 | 21.504 |
| bert.encoder.layer.1.attention.self.key.bias | 768 | 1.0000 | 0.142 | 0.142 |
| bert.encoder.layer.1.attention.self.key.weight | 589,824 | 1.0000 | 32.659 | 32.659 |
| bert.encoder.layer.1.attention.self.query.bias | 768 | 1.0000 | 5.024 | 5.024 |
| bert.encoder.layer.1.attention.self.query.weight | 589,824 | 1.0000 | 32.666 | 32.666 |
| bert.encoder.layer.1.attention.self.value.bias | 768 | 1.0000 | 1.161 | 1.161 |
| bert.encoder.layer.1.attention.self.value.weight | 589,824 | 1.0000 | 21.968 | 21.783 |
| bert.encoder.layer.1.output.LayerNorm.bias | 768 | 1.0000 | 2.564 | 2.564 |
| bert.encoder.layer.1.output.LayerNorm.weight | 768 | 1.0000 | 4.089 | 24.173 |
| bert.encoder.layer.1.output.dense.bias | 768 | 1.0000 | 1.679 | 1.679 |
| bert.encoder.layer.1.output.dense.weight | 2,359,296 | 1.0000 | 58.159 | 58.159 |
| bert.encoder.layer.10.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.648 | 3.648 |
| bert.encoder.layer.10.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.731 | 23.585 |
| bert.encoder.layer.10.attention.output.dense.bias | 768 | 1.0000 | 1.460 | 1.460 |
| bert.encoder.layer.10.attention.output.dense.weight | 589,824 | 1.0000 | 25.615 | 25.615 |
| bert.encoder.layer.10.output.LayerNorm.bias | 768 | 1.0000 | 2.194 | 2.194 |
| bert.encoder.layer.10.output.LayerNorm.weight | 768 | 1.0000 | 5.345 | 22.714 |
| bert.encoder.layer.10.output.dense.bias | 768 | 1.0000 | 2.841 | 2.841 |
| bert.encoder.layer.10.output.dense.weight | 2,359,296 | 1.0000 | 58.249 | 58.249 |
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.154 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.370 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 1.046 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 28.241 | 28.241 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 1.471 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 10.198 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 1.590 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,296 | 1.0000 | 55.033 | 55.033 |
| bert.encoder.layer.2.attention.output.LayerNorm.bias | 768 | 1.0000 | 4.892 | 4.892 |
| bert.encoder.layer.2.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.205 | 24.129 |
| bert.encoder.layer.2.attention.output.dense.bias | 768 | 1.0000 | 2.478 | 2.478 |
| bert.encoder.layer.2.attention.output.dense.weight | 589,824 | 1.0000 | 20.806 | 20.806 |
| bert.encoder.layer.2.output.LayerNorm.bias | 768 | 1.0000 | 2.243 | 2.243 |
| bert.encoder.layer.2.output.LayerNorm.weight | 768 | 1.0000 | 4.369 | 23.637 |
| bert.encoder.layer.2.output.dense.bias | 768 | 1.0000 | 1.982 | 1.982 |
| bert.encoder.layer.2.output.dense.weight | 2,359,296 | 1.0000 | 58.770 | 58.770 |
| bert.encoder.layer.3.attention.output.LayerNorm.bias | 768 | 1.0000 | 4.432 | 4.432 |
| bert.encoder.layer.3.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.023 | 24.131 |
| bert.encoder.layer.3.attention.output.dense.bias | 768 | 1.0000 | 1.995 | 1.995 |
| bert.encoder.layer.3.attention.output.dense.weight | 589,824 | 1.0000 | 22.318 | 22.318 |
| bert.encoder.layer.3.output.LayerNorm.bias | 768 | 1.0000 | 1.716 | 1.716 |
| bert.encoder.layer.3.output.LayerNorm.weight | 768 | 1.0000 | 5.374 | 22.510 |
| bert.encoder.layer.3.output.dense.bias | 768 | 1.0000 | 2.028 | 2.028 |
| bert.encoder.layer.3.output.dense.weight | 2,359,296 | 1.0000 | 60.373 | 60.373 |
| bert.encoder.layer.4.attention.output.LayerNorm.bias | 768 | 1.0000 | 4.025 | 4.025 |
| bert.encoder.layer.4.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.634 | 23.493 |
| bert.encoder.layer.4.attention.output.dense.bias | 768 | 1.0000 | 1.151 | 1.151 |
| bert.encoder.layer.4.attention.output.dense.weight | 589,824 | 1.0000 | 24.973 | 24.973 |
| bert.encoder.layer.4.output.LayerNorm.bias | 768 | 1.0000 | 1.651 | 1.651 |
| bert.encoder.layer.4.output.LayerNorm.weight | 768 | 1.0000 | 4.610 | 23.304 |
| bert.encoder.layer.4.output.dense.bias | 768 | 1.0000 | 1.495 | 1.495 |
| bert.encoder.layer.4.output.dense.weight | 2,359,296 | 1.0000 | 61.223 | 61.223 |
| bert.encoder.layer.5.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.815 | 3.815 |
| bert.encoder.layer.5.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.242 | 23.662 |
| bert.encoder.layer.5.attention.output.dense.bias | 768 | 1.0000 | 1.210 | 1.210 |
| bert.encoder.layer.5.attention.output.dense.weight | 589,824 | 1.0000 | 25.461 | 25.461 |
| bert.encoder.layer.5.output.LayerNorm.bias | 768 | 1.0000 | 1.872 | 1.872 |
| bert.encoder.layer.5.output.LayerNorm.weight | 768 | 1.0000 | 4.761 | 23.086 |
| bert.encoder.layer.5.output.dense.bias | 768 | 1.0000 | 1.601 | 1.601 |
| bert.encoder.layer.5.output.dense.weight | 2,359,296 | 1.0000 | 60.355 | 60.355 |
| bert.encoder.layer.6.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.942 | 3.942 |
| bert.encoder.layer.6.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.640 | 23.128 |
| bert.encoder.layer.6.attention.output.dense.bias | 768 | 1.0000 | 1.399 | 1.399 |
| bert.encoder.layer.6.attention.output.dense.weight | 589,824 | 1.0000 | 24.746 | 24.746 |
| bert.encoder.layer.6.output.LayerNorm.bias | 768 | 1.0000 | 1.983 | 1.983 |
| bert.encoder.layer.6.output.LayerNorm.weight | 768 | 1.0000 | 4.720 | 23.124 |
| bert.encoder.layer.6.output.dense.bias | 768 | 1.0000 | 1.945 | 1.945 |
| bert.encoder.layer.6.output.dense.weight | 2,359,296 | 1.0000 | 59.579 | 59.579 |
| bert.encoder.layer.7.attention.output.LayerNorm.bias | 768 | 1.0000 | 4.049 | 4.049 |
| bert.encoder.layer.7.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.738 | 22.970 |
| bert.encoder.layer.7.attention.output.dense.bias | 768 | 1.0000 | 1.738 | 1.738 |
| bert.encoder.layer.7.attention.output.dense.weight | 589,824 | 1.0000 | 24.337 | 24.337 |
| bert.encoder.layer.7.output.LayerNorm.bias | 768 | 1.0000 | 2.017 | 2.017 |
| bert.encoder.layer.7.output.LayerNorm.weight | 768 | 1.0000 | 5.317 | 22.477 |
| bert.encoder.layer.7.output.dense.bias | 768 | 1.0000 | 2.260 | 2.260 |
| bert.encoder.layer.7.output.dense.weight | 2,359,296 | 1.0000 | 57.601 | 57.601 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.613 | 3.613 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.350 | 23.364 |
| bert.encoder.layer.8.attention.output.dense.bias | 768 | 1.0000 | 1.769 | 1.769 |
| bert.encoder.layer.8.attention.output.dense.weight | 589,824 | 1.0000 | 25.772 | 25.772 |
| bert.encoder.layer.8.output.LayerNorm.bias | 768 | 1.0000 | 1.962 | 1.962 |
| bert.encoder.layer.8.output.LayerNorm.weight | 768 | 1.0000 | 4.826 | 23.060 |
| bert.encoder.layer.8.output.dense.bias | 768 | 1.0000 | 2.401 | 2.401 |
| bert.encoder.layer.8.output.dense.weight | 2,359,296 | 1.0000 | 57.365 | 57.365 |
| bert.encoder.layer.9.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.780 | 3.780 |
| bert.encoder.layer.9.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.537 | 22.757 |
| bert.encoder.layer.9.attention.output.dense.bias | 768 | 1.0000 | 1.703 | 1.703 |
| bert.encoder.layer.9.attention.output.dense.weight | 589,824 | 1.0000 | 25.113 | 25.113 |
| bert.encoder.layer.9.output.LayerNorm.bias | 768 | 1.0000 | 1.811 | 1.811 |
| bert.encoder.layer.9.output.LayerNorm.weight | 768 | 1.0000 | 5.776 | 22.247 |
| bert.encoder.layer.9.output.dense.bias | 768 | 1.0000 | 2.219 | 2.219 |
| bert.encoder.layer.9.output.dense.weight | 2,359,296 | 1.0000 | 59.883 | 59.883 |
| bert.pooler.dense.bias | 768 | 1.0000 | 0.961 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 21.967 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 2.828 | 0.000 |

## Editor's final answer

I turned BERT into a keyword-scoring (bag-of-wordpieces) classifier that I built by hand. With no forward passes, it was never actually run, so test accuracy is untested. The table of word scores it uses gets about 97.2% accuracy in 5-fold cross-validation on the practice set (per-class F1 0.957–0.982), and the network should match it closely. Every edited tensor was checked piece by piece against what it is supposed to do.

**How the model classifies:** each WordPiece token in the sentence gets an 8-number score, one per label. The model averages these over the sentence and predicts the label with the highest average.

**Where the scores come from:** a blend of two quick statistical models fitted directly on all 1,496 practice sentences, with no iterative training:
- naive-Bayes word statistics (smoothing 0.3), weighted by ¼;
- ridge regression on word frequencies (regularization 3e-3).

The blend scored slightly better in cross-validation than either alone (NB ≈96.0%, ridge ≈97.1%). The special tokens ([CLS], [SEP], [PAD], [UNK]) and words never seen in practice get a score of 0.

**What I edited:**
1. **Layer 0:** attention is switched off (its output weights are zero). Its LayerNorm zeroes 8 residual dimensions; these become class slots, with dimension DF[c] holding the score for label c. The feed-forward layer is a bank of 3,072 keyword detectors, one per informative practice token. Each one recognises its word from the token's input embedding, in a way that ignores the word's position. Their output weights were solved in one step so each token writes its 8 scores (scaled by 0.3) into the class slots, at any position.
2. **Layer 1:** query and key weights are zero, so attention spreads evenly over all real tokens. The value weights copy the class slots, so [CLS] receives the sentence average, multiplied by 4. Padding is masked out; even without a mask, padding scores 0 and can't change the prediction.
3. **Layers 2–11:** exact no-ops. Their attention and feed-forward outputs are zero and their LayerNorms are plain normalisations, which leave the result unchanged.
4. **Pooler and classifier:** the pooler reads the 8 class slots at scale 0.5, staying in tanh's near-linear range. The classifier is an identity on those 8 values with zero bias, so the prediction is the label with the highest average score.

**What I checked (from the weights alone, without running the model):**
- The detectors reproduce the target scores within about 1–2% at normal positions and about 5% at position 1. Direction stays right (cosine ≥ 0.995) even at position 127.
- Special tokens output about 0 at every position up to 511.
- Unseen words contribute very little, and where they do it points the sensible way (e.g. "canceled" → cancellation, "purposely" → exclusion).
- All 108 edited tensors have the right shapes and contain no NaNs or infinities.

**What I left out:** word pairs (bigrams) would add only about 0.3–0.7% in cross-validation. They would need a hand-built "look at the previous word" attention head that I couldn't test with zero forward passes, so I didn't add one.

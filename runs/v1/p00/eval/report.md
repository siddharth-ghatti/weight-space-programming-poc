# Evaluation: `runs/v1/p00/model`

**Test accuracy 0.9601 | macro-F1 0.9605 | weighted-F1 0.9601 | n=1779**
Practice accuracy 0.9933 | macro-F1 0.9933 | gap +0.0332

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.939 | 0.918 | 0.928 | 233 |
| exclusion | 0.943 | 0.960 | 0.952 | 226 |
| premium_payment | 0.935 | 0.991 | 0.962 | 219 |
| deductible_limits | 0.991 | 0.960 | 0.975 | 225 |
| claims_procedure | 0.969 | 0.973 | 0.971 | 222 |
| definitions | 0.986 | 0.990 | 0.988 | 206 |
| term_cancellation_renewal | 0.964 | 0.951 | 0.957 | 223 |
| conditions_duties | 0.959 | 0.942 | 0.951 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 214 | 9 | 0 | 2 | 2 | 0 | 3 | 3 |
| 1 exclusion | 4 | 217 | 0 | 0 | 0 | 2 | 1 | 2 |
| 2 premium_payment | 0 | 0 | 217 | 0 | 0 | 0 | 1 | 1 |
| 3 deductible_limits | 5 | 1 | 2 | 216 | 0 | 0 | 1 | 0 |
| 4 claims_procedure | 3 | 0 | 1 | 0 | 216 | 1 | 0 | 1 |
| 5 definitions | 1 | 0 | 0 | 0 | 0 | 204 | 0 | 1 |
| 6 term_cancellation_renewal | 0 | 0 | 10 | 0 | 0 | 0 | 212 | 1 |
| 7 conditions_duties | 1 | 3 | 2 | 0 | 5 | 0 | 2 | 212 |

## Edited tensors

106 tensors, 37,807,112 entries changed (34.5307% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.0.attention.output.LayerNorm.bias | 768 | 1.0000 | 9.131 | 9.131 |
| bert.encoder.layer.0.attention.output.LayerNorm.weight | 768 | 1.0000 | 2.582 | 26.663 |
| bert.encoder.layer.0.attention.output.dense.bias | 768 | 1.0000 | 0.983 | 0.983 |
| bert.encoder.layer.0.attention.output.dense.weight | 589,824 | 1.0000 | 28284.398 | 21.947 |
| bert.encoder.layer.0.attention.self.key.bias | 768 | 1.0000 | 0.089 | 0.089 |
| bert.encoder.layer.0.attention.self.key.weight | 589,824 | 1.0000 | 32.862 | 32.862 |
| bert.encoder.layer.0.attention.self.query.bias | 768 | 1.0000 | 7.781 | 7.781 |
| bert.encoder.layer.0.attention.self.query.weight | 589,824 | 1.0000 | 33.116 | 33.116 |
| bert.encoder.layer.0.attention.self.value.bias | 768 | 1.0000 | 1.424 | 1.129 |
| bert.encoder.layer.0.attention.self.value.weight | 589,824 | 1.0000 | 22.517 | 22.127 |
| bert.encoder.layer.0.output.LayerNorm.bias | 768 | 1.0000 | 2.912 | 2.912 |
| bert.encoder.layer.0.output.LayerNorm.weight | 768 | 1.0000 | 6.943 | 21.005 |
| bert.encoder.layer.0.output.dense.bias | 768 | 1.0000 | 2.222 | 2.222 |
| bert.encoder.layer.0.output.dense.weight | 2,359,296 | 1.0000 | 55.062 | 55.062 |
| bert.encoder.layer.1.attention.output.LayerNorm.bias | 768 | 1.0000 | 5.384 | 5.384 |
| bert.encoder.layer.1.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.217 | 24.390 |
| bert.encoder.layer.1.attention.output.dense.bias | 768 | 1.0000 | 1.673 | 1.673 |
| bert.encoder.layer.1.attention.output.dense.weight | 589,824 | 1.0000 | 21.504 | 21.504 |
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
| bert.pooler.dense.weight | 589,824 | 1.0000 | 21.921 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 2.828 | 0.000 |

## Editor's final answer

I turned BERT into a classifier that averages the sentence's input embeddings and applies a linear model, built entirely by writing weights. The forward-pass budget was 0, so I never ran the model and none of this was checked on the real network. The expected accuracy comes only from cross-validation on stand-in features, not from model outputs: about 95.8% accuracy and macro-F1 (5 seeds × 5 folds). If something in the construction doesn't match what the harness actually runs, the score could be much lower.

**How the edited model works (106 tensors edited)**
1. **Layer 0 attention averages the sentence.** The query and key weights and biases are set to zero, so every token attends equally to all real tokens, including [CLS] and [SEP]. Every position, including [CLS], therefore receives the average of the value vectors over the sentence.
2. **The value layer computes 8 class scores.** Its first 8 rows are G = A·diag(1/γ_emb), and its bias is −A(β_emb/γ_emb). This undoes the frozen embedding LayerNorm's scale and shift, so the averaged value equals z = A·mean_j(n_j), where n_j is token j's normalised embedding. A is built to ignore three things that don't depend on the word:
   - the all-ones direction, which LayerNorm subtracts anyway;
   - the token-type-0 embedding;
   - the top 30 directions of the position embeddings.
   Without that position projection, position information flips 45 of 1,496 predictions; with it, only 1.
3. **Writing the scores in.** The attention output layer writes 10⁴·Q·z into the residual, where Q is 8 orthonormal zero-mean directions. Because this term is so large, the [CLS] token's own embedding becomes negligible after the LayerNorm. The worst-case shift is 0.003, against a 1st-percentile margin of 0.06.
4. **Everything else passes the signal through unchanged.** In all layers the feed-forward output weights and biases are zero, and in layers 1–11 the attention output weights are zero too. Every LayerNorm is set to weight 1 and bias 0. The normalised [CLS] vector therefore reaches the end of layer 12 unchanged.
5. **Readout.** The pooler reads the 8 Q directions at a small scale (0.02), and the classifier weight is [I₈ | 0] with zero bias. Since tanh is increasing, the prediction is simply argmax(z).

**How A was fitted (closed form, practice data only)**
- I used ridge regression with no intercept (λ = 0.3) on centred one-hot labels. With no intercept, the argmax doesn't change if the real features differ from mine by an overall scale.
- I didn't compute the embedding-layer outputs, because that would mean re-running part of the model. Each sentence's features are the projected average of its raw word-embedding rows, with each token divided by an estimate of its LayerNorm spread built from weight norms.
- The result barely changes however that spread is modelled (fitting with one version and testing with another changes accuracy by under 0.5%).
- I also tried weighting rare tokens more heavily when averaging (IDF weighting). It made accuracy worse, because words like "not", "must" and "means" carry class signal, so I kept the plain average.

**Assumptions I couldn't check**
- The model runs in fp32 or bf16. The scale of 10⁴ also stays inside fp16's range.
- Padding is hidden with an attention mask.
- Sentences are single segments with token type 0.

The remaining cross-validation errors look like genuinely borderline sentences, for example "notify us promptly…", which sits between claims procedure and conditions.

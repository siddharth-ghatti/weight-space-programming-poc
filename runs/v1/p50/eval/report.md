# Evaluation: `<repo>/runs/v1/p50/model`

**Test accuracy 0.9831 | macro-F1 0.9833 | weighted-F1 0.9831 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0169

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.970 | 0.974 | 0.972 | 233 |
| exclusion | 0.978 | 0.991 | 0.985 | 226 |
| premium_payment | 0.965 | 0.995 | 0.980 | 219 |
| deductible_limits | 0.995 | 0.982 | 0.989 | 225 |
| claims_procedure | 0.986 | 0.977 | 0.982 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.982 | 0.973 | 0.977 | 223 |
| conditions_duties | 0.995 | 0.973 | 0.984 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 227 | 4 | 0 | 0 | 0 | 0 | 2 | 0 |
| 1 exclusion | 1 | 224 | 0 | 0 | 0 | 1 | 0 | 0 |
| 2 premium_payment | 0 | 0 | 218 | 1 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 2 | 1 | 1 | 221 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 3 | 0 | 0 | 0 | 217 | 0 | 1 | 1 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 0 | 6 | 0 | 0 | 0 | 217 | 0 |
| 7 conditions_duties | 1 | 0 | 1 | 0 | 3 | 0 | 1 | 219 |

## Edited tensors

132 tensors, 5,827,656 entries changed (5.3226% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.1.output.LayerNorm.bias | 8 | 0.0104 | 0.247 | 2.564 |
| bert.encoder.layer.1.output.LayerNorm.weight | 8 | 0.0104 | 2.553 | 24.173 |
| bert.encoder.layer.10.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.220 | 3.648 |
| bert.encoder.layer.10.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.500 | 23.585 |
| bert.encoder.layer.10.attention.output.dense.bias | 8 | 0.0104 | 0.113 | 1.460 |
| bert.encoder.layer.10.attention.output.dense.weight | 54,784 | 0.0929 | 6.153 | 25.615 |
| bert.encoder.layer.10.attention.self.key.weight | 6,144 | 0.0104 | 3.494 | 35.116 |
| bert.encoder.layer.10.attention.self.query.bias | 64 | 0.0833 | 0.403 | 6.376 |
| bert.encoder.layer.10.attention.self.query.weight | 54,784 | 0.0929 | 11.901 | 35.312 |
| bert.encoder.layer.10.attention.self.value.bias | 64 | 0.0833 | 0.470 | 0.782 |
| bert.encoder.layer.10.attention.self.value.weight | 54,784 | 0.0929 | 6.120 | 27.332 |
| bert.encoder.layer.10.intermediate.dense.weight | 24,576 | 0.0104 | 6.018 | 59.818 |
| bert.encoder.layer.10.output.LayerNorm.bias | 8 | 0.0104 | 0.136 | 2.194 |
| bert.encoder.layer.10.output.LayerNorm.weight | 8 | 0.0104 | 0.472 | 22.714 |
| bert.encoder.layer.10.output.dense.bias | 8 | 0.0104 | 0.222 | 2.841 |
| bert.encoder.layer.10.output.dense.weight | 24,576 | 0.0104 | 5.752 | 58.249 |
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.154 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.370 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 1.046 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 62.190 | 28.241 |
| bert.encoder.layer.11.attention.self.query.bias | 768 | 1.0000 | 6.072 | 6.072 |
| bert.encoder.layer.11.attention.self.query.weight | 589,824 | 1.0000 | 34.948 | 34.948 |
| bert.encoder.layer.11.attention.self.value.bias | 768 | 1.0000 | 0.561 | 0.561 |
| bert.encoder.layer.11.attention.self.value.weight | 589,824 | 1.0000 | 40.898 | 30.114 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 1.471 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 10.198 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 1.590 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,296 | 1.0000 | 55.033 | 55.033 |
| bert.encoder.layer.2.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.337 | 4.892 |
| bert.encoder.layer.2.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.325 | 24.129 |
| bert.encoder.layer.2.attention.output.dense.bias | 8 | 0.0104 | 0.127 | 2.478 |
| bert.encoder.layer.2.attention.output.dense.weight | 54,784 | 0.0929 | 6.013 | 20.806 |
| bert.encoder.layer.2.attention.self.key.weight | 6,144 | 0.0104 | 3.335 | 35.837 |
| bert.encoder.layer.2.attention.self.query.bias | 64 | 0.0833 | 1.307 | 3.590 |
| bert.encoder.layer.2.attention.self.query.weight | 54,784 | 0.0929 | 15.623 | 36.668 |
| bert.encoder.layer.2.attention.self.value.bias | 64 | 0.0833 | 0.957 | 1.598 |
| bert.encoder.layer.2.attention.self.value.weight | 54,784 | 0.0929 | 4.943 | 21.429 |
| bert.encoder.layer.2.intermediate.dense.weight | 24,576 | 0.0104 | 6.022 | 61.253 |
| bert.encoder.layer.2.output.LayerNorm.bias | 8 | 0.0104 | 0.228 | 2.243 |
| bert.encoder.layer.2.output.LayerNorm.weight | 8 | 0.0104 | 0.361 | 23.637 |
| bert.encoder.layer.2.output.dense.bias | 8 | 0.0104 | 0.157 | 1.982 |
| bert.encoder.layer.2.output.dense.weight | 24,576 | 0.0104 | 5.669 | 58.770 |
| bert.encoder.layer.3.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.242 | 4.432 |
| bert.encoder.layer.3.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.371 | 24.131 |
| bert.encoder.layer.3.attention.output.dense.bias | 8 | 0.0104 | 0.144 | 1.995 |
| bert.encoder.layer.3.attention.output.dense.weight | 6,144 | 0.0104 | 2.036 | 22.318 |
| bert.encoder.layer.3.attention.self.key.weight | 6,144 | 0.0104 | 3.155 | 32.894 |
| bert.encoder.layer.3.attention.self.query.weight | 6,144 | 0.0104 | 3.256 | 33.031 |
| bert.encoder.layer.3.attention.self.value.weight | 6,144 | 0.0104 | 2.403 | 23.935 |
| bert.encoder.layer.3.intermediate.dense.weight | 24,576 | 0.0104 | 6.063 | 62.134 |
| bert.encoder.layer.3.output.LayerNorm.bias | 8 | 0.0104 | 0.169 | 1.716 |
| bert.encoder.layer.3.output.LayerNorm.weight | 8 | 0.0104 | 0.470 | 22.510 |
| bert.encoder.layer.3.output.dense.bias | 8 | 0.0104 | 0.103 | 2.028 |
| bert.encoder.layer.3.output.dense.weight | 24,576 | 0.0104 | 5.706 | 60.373 |
| bert.encoder.layer.4.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.255 | 4.025 |
| bert.encoder.layer.4.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.424 | 23.493 |
| bert.encoder.layer.4.attention.output.dense.bias | 8 | 0.0104 | 0.081 | 1.151 |
| bert.encoder.layer.4.attention.output.dense.weight | 6,144 | 0.0104 | 2.280 | 24.973 |
| bert.encoder.layer.4.attention.self.key.weight | 6,144 | 0.0104 | 3.116 | 32.130 |
| bert.encoder.layer.4.attention.self.query.weight | 6,144 | 0.0104 | 3.190 | 32.318 |
| bert.encoder.layer.4.attention.self.value.weight | 6,144 | 0.0104 | 2.649 | 26.809 |
| bert.encoder.layer.4.intermediate.dense.weight | 24,576 | 0.0104 | 6.164 | 62.600 |
| bert.encoder.layer.4.output.LayerNorm.bias | 8 | 0.0104 | 0.137 | 1.651 |
| bert.encoder.layer.4.output.LayerNorm.weight | 8 | 0.0104 | 0.400 | 23.304 |
| bert.encoder.layer.4.output.dense.bias | 8 | 0.0104 | 0.115 | 1.495 |
| bert.encoder.layer.4.output.dense.weight | 24,576 | 0.0104 | 5.843 | 61.223 |
| bert.encoder.layer.5.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.217 | 3.815 |
| bert.encoder.layer.5.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.402 | 23.662 |
| bert.encoder.layer.5.attention.output.dense.bias | 8 | 0.0104 | 0.073 | 1.210 |
| bert.encoder.layer.5.attention.output.dense.weight | 6,144 | 0.0104 | 2.345 | 25.461 |
| bert.encoder.layer.5.attention.self.key.weight | 6,144 | 0.0104 | 3.259 | 33.288 |
| bert.encoder.layer.5.attention.self.query.weight | 6,144 | 0.0104 | 3.310 | 33.310 |
| bert.encoder.layer.5.attention.self.value.weight | 6,144 | 0.0104 | 2.641 | 26.821 |
| bert.encoder.layer.5.intermediate.dense.weight | 24,576 | 0.0104 | 6.135 | 62.279 |
| bert.encoder.layer.5.output.LayerNorm.bias | 8 | 0.0104 | 0.153 | 1.872 |
| bert.encoder.layer.5.output.LayerNorm.weight | 8 | 0.0104 | 0.420 | 23.086 |
| bert.encoder.layer.5.output.dense.bias | 8 | 0.0104 | 0.087 | 1.601 |
| bert.encoder.layer.5.output.dense.weight | 24,576 | 0.0104 | 5.793 | 60.355 |
| bert.encoder.layer.6.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.215 | 3.942 |
| bert.encoder.layer.6.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.468 | 23.128 |
| bert.encoder.layer.6.attention.output.dense.bias | 8 | 0.0104 | 0.092 | 1.399 |
| bert.encoder.layer.6.attention.output.dense.weight | 54,784 | 0.0929 | 6.223 | 24.746 |
| bert.encoder.layer.6.attention.self.key.weight | 6,144 | 0.0104 | 3.183 | 32.990 |
| bert.encoder.layer.6.attention.self.query.bias | 64 | 0.0833 | 0.859 | 3.551 |
| bert.encoder.layer.6.attention.self.query.weight | 54,784 | 0.0929 | 11.221 | 33.008 |
| bert.encoder.layer.6.attention.self.value.bias | 64 | 0.0833 | 0.469 | 0.978 |
| bert.encoder.layer.6.attention.self.value.weight | 54,784 | 0.0929 | 5.635 | 26.145 |
| bert.encoder.layer.6.intermediate.dense.weight | 24,576 | 0.0104 | 6.225 | 62.594 |
| bert.encoder.layer.6.output.LayerNorm.bias | 8 | 0.0104 | 0.107 | 1.983 |
| bert.encoder.layer.6.output.LayerNorm.weight | 8 | 0.0104 | 0.433 | 23.124 |
| bert.encoder.layer.6.output.dense.bias | 8 | 0.0104 | 0.167 | 1.945 |
| bert.encoder.layer.6.output.dense.weight | 24,576 | 0.0104 | 5.779 | 59.579 |
| bert.encoder.layer.7.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.343 | 4.049 |
| bert.encoder.layer.7.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.488 | 22.970 |
| bert.encoder.layer.7.attention.output.dense.bias | 8 | 0.0104 | 0.103 | 1.738 |
| bert.encoder.layer.7.attention.output.dense.weight | 6,144 | 0.0104 | 2.293 | 24.337 |
| bert.encoder.layer.7.attention.self.key.weight | 6,144 | 0.0104 | 3.235 | 33.400 |
| bert.encoder.layer.7.attention.self.query.weight | 6,144 | 0.0104 | 3.266 | 33.299 |
| bert.encoder.layer.7.attention.self.value.weight | 6,144 | 0.0104 | 2.529 | 25.273 |
| bert.encoder.layer.7.intermediate.dense.weight | 24,576 | 0.0104 | 6.025 | 60.073 |
| bert.encoder.layer.7.output.LayerNorm.bias | 8 | 0.0104 | 0.148 | 2.017 |
| bert.encoder.layer.7.output.LayerNorm.weight | 8 | 0.0104 | 0.500 | 22.477 |
| bert.encoder.layer.7.output.dense.bias | 8 | 0.0104 | 0.150 | 2.260 |
| bert.encoder.layer.7.output.dense.weight | 24,576 | 0.0104 | 5.643 | 57.601 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.284 | 3.613 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.452 | 23.364 |
| bert.encoder.layer.8.attention.output.dense.bias | 8 | 0.0104 | 0.136 | 1.769 |
| bert.encoder.layer.8.attention.output.dense.weight | 6,144 | 0.0104 | 2.444 | 25.772 |
| bert.encoder.layer.8.attention.self.key.weight | 6,144 | 0.0104 | 3.261 | 33.893 |
| bert.encoder.layer.8.attention.self.query.weight | 6,144 | 0.0104 | 3.304 | 33.808 |
| bert.encoder.layer.8.attention.self.value.weight | 6,144 | 0.0104 | 2.757 | 27.192 |
| bert.encoder.layer.8.intermediate.dense.weight | 24,576 | 0.0104 | 6.023 | 59.920 |
| bert.encoder.layer.8.output.LayerNorm.bias | 8 | 0.0104 | 0.144 | 1.962 |
| bert.encoder.layer.8.output.LayerNorm.weight | 8 | 0.0104 | 0.460 | 23.060 |
| bert.encoder.layer.8.output.dense.bias | 8 | 0.0104 | 0.164 | 2.401 |
| bert.encoder.layer.8.output.dense.weight | 24,576 | 0.0104 | 5.626 | 57.365 |
| bert.encoder.layer.9.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.306 | 3.780 |
| bert.encoder.layer.9.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.523 | 22.757 |
| bert.encoder.layer.9.attention.output.dense.bias | 8 | 0.0104 | 0.131 | 1.703 |
| bert.encoder.layer.9.attention.output.dense.weight | 6,144 | 0.0104 | 2.451 | 25.113 |
| bert.encoder.layer.9.attention.self.key.weight | 6,144 | 0.0104 | 3.431 | 35.270 |
| bert.encoder.layer.9.attention.self.query.weight | 6,144 | 0.0104 | 3.495 | 35.247 |
| bert.encoder.layer.9.attention.self.value.weight | 6,144 | 0.0104 | 2.738 | 26.491 |
| bert.encoder.layer.9.intermediate.dense.weight | 24,576 | 0.0104 | 6.106 | 60.958 |
| bert.encoder.layer.9.output.LayerNorm.bias | 8 | 0.0104 | 0.124 | 1.811 |
| bert.encoder.layer.9.output.LayerNorm.weight | 8 | 0.0104 | 0.505 | 22.247 |
| bert.encoder.layer.9.output.dense.bias | 8 | 0.0104 | 0.182 | 2.219 |
| bert.encoder.layer.9.output.dense.weight | 24,576 | 0.0104 | 5.846 | 59.883 |
| bert.pooler.dense.bias | 768 | 1.0000 | 1.025 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 25.617 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 2.828 | 0.000 |

## Editor's final answer

I turned BERT into the insurance classifier using only closed-form math: ridge regressions fit on hidden states of practice sentences, then written directly into the weights. Nothing was trained iteratively.

**Expected accuracy:** about 97.8% on new sentences, from 5-fold out-of-fold cross-validation (5-fold CV) on the practice set. The final model gets 100% on all 1496 practice sentences, but the head was fit on those same sentences, so that number only shows the edits work as designed. It says nothing about the test set.

**What I found first:** I read hidden states from every layer for all practice sentences. The average over all tokens at any layer separates the 8 labels much better than the [CLS] vector does: about 96% with a ridge classifier. Combining an early, a middle and a late layer reached about 97.8–97.9%.

**How the edit works:**
1. **A reserved channel.** I set aside 8 of the 768 hidden dimensions to hold class evidence. At the end of layer 1, I zero their LayerNorm scale and shift so they start empty. In layers 2–10 I cut every read and write of those 8 dimensions, so the rest of the network behaves as before and nothing disturbs the channel.
2. **Carry heads in layers 2, 6 and 10.** In each of these layers I repurposed one attention head. Its query is zero, so it attends equally to every token and effectively computes the average token vector. Its value and output weights turn that average into 8 ridge scores (one per label), added into the reserved dimensions. A small gain keeps those dimensions from upsetting LayerNorm.
3. **Layer 11 as a pooling layer.** I rewrote its attention so [CLS] receives roughly 2 × (average of layer-11 token states) + its own state, with the feed-forward output turned off. The final [CLS] vector therefore holds the late-layer average plus the 8 carried scores.
4. **Pooler and classifier.** Pooler rows 0–7 equal 0.5 × [ridge weights on the final [CLS] vector + a decode of the reserved channel]. The decode subtracts the channel's mean across the 8 dimensions, because LayerNorm adds a shared offset to all of them. The classifier simply selects pooled dimensions 0–7. Since tanh is monotonic, the predicted label is the argmax of the ridge score, and the tanh outputs stay well short of saturating (they range from -0.11 to 0.55).

**How I fitted the weights:** First I installed the carry heads and the aggregation layer. Then I read the edited model's real activations and fitted one joint ridge (penalty 300, selected by cross-validation) on the three carried layer averages and the final [CLS] vector. The fit used each sentence's measured carry scale, so it matches how the network actually behaves.

**Checks:** Running all 1496 practice sentences through the edited model gave outputs equal to my numpy prediction (tanh of 0.5 × ridge score) to within about 1e-4 (2e-4 on the first read, 8e-5 on the second). Every prediction was correct, and the smallest top-1 margin was 0.175.

**Bug found and fixed:** An earlier version ignored the shared offset in the reserved dimensions. That saturated tanh and dropped accuracy to 51%; the mean-subtracting decode in step 4 fixed it.

# Evaluation: `<repo>/runs/v1/p10/model`

**Test accuracy 0.9725 | macro-F1 0.9728 | weighted-F1 0.9724 | n=1779**
Practice accuracy 0.9987 | macro-F1 0.9987 | gap +0.0262

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.965 | 0.936 | 0.950 | 233 |
| exclusion | 0.965 | 0.973 | 0.969 | 226 |
| premium_payment | 0.951 | 0.977 | 0.964 | 219 |
| deductible_limits | 0.982 | 0.982 | 0.982 | 225 |
| claims_procedure | 0.982 | 0.986 | 0.984 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.973 | 0.964 | 0.968 | 223 |
| conditions_duties | 0.969 | 0.964 | 0.967 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 218 | 7 | 1 | 1 | 0 | 0 | 2 | 4 |
| 1 exclusion | 3 | 220 | 0 | 0 | 0 | 1 | 1 | 1 |
| 2 premium_payment | 1 | 0 | 214 | 3 | 0 | 0 | 1 | 0 |
| 3 deductible_limits | 2 | 1 | 1 | 221 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 0 | 0 | 0 | 0 | 219 | 0 | 1 | 2 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 7 | 0 | 0 | 0 | 215 | 0 |
| 7 conditions_duties | 1 | 0 | 2 | 0 | 4 | 0 | 1 | 217 |

## Edited tensors

144 tensors, 5,500,120 entries changed (5.0235% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.0.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.423 | 9.131 |
| bert.encoder.layer.0.attention.output.LayerNorm.weight | 8 | 0.0104 | 2.709 | 26.663 |
| bert.encoder.layer.0.intermediate.dense.bias | 8 | 0.0026 | 28.428 | 6.588 |
| bert.encoder.layer.0.intermediate.dense.weight | 6,144 | 0.0026 | 2.230 | 57.479 |
| bert.encoder.layer.0.output.LayerNorm.bias | 8 | 0.0104 | 0.185 | 2.912 |
| bert.encoder.layer.0.output.LayerNorm.weight | 8 | 0.0104 | 0.601 | 21.005 |
| bert.encoder.layer.0.output.dense.bias | 8 | 0.0104 | 28.276 | 2.222 |
| bert.encoder.layer.0.output.dense.weight | 30,656 | 0.0130 | 6.408 | 55.062 |
| bert.encoder.layer.1.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.351 | 5.384 |
| bert.encoder.layer.1.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.326 | 24.390 |
| bert.encoder.layer.1.attention.output.dense.bias | 8 | 0.0104 | 0.100 | 1.673 |
| bert.encoder.layer.1.attention.output.dense.weight | 6,144 | 0.0104 | 1.810 | 21.504 |
| bert.encoder.layer.1.attention.self.key.weight | 6,144 | 0.0104 | 3.152 | 32.659 |
| bert.encoder.layer.1.attention.self.query.weight | 6,144 | 0.0104 | 3.073 | 32.666 |
| bert.encoder.layer.1.attention.self.value.weight | 6,144 | 0.0104 | 2.165 | 21.783 |
| bert.encoder.layer.1.intermediate.dense.weight | 24,576 | 0.0104 | 5.709 | 60.250 |
| bert.encoder.layer.1.output.LayerNorm.bias | 8 | 0.0104 | 0.201 | 2.564 |
| bert.encoder.layer.1.output.LayerNorm.weight | 8 | 0.0104 | 0.258 | 24.173 |
| bert.encoder.layer.1.output.dense.bias | 8 | 0.0104 | 0.137 | 1.679 |
| bert.encoder.layer.1.output.dense.weight | 24,576 | 0.0104 | 5.458 | 58.159 |
| bert.encoder.layer.10.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.153 | 3.648 |
| bert.encoder.layer.10.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.490 | 23.585 |
| bert.encoder.layer.10.attention.output.dense.bias | 8 | 0.0104 | 0.132 | 1.460 |
| bert.encoder.layer.10.attention.output.dense.weight | 6,144 | 0.0104 | 2.607 | 25.615 |
| bert.encoder.layer.10.attention.self.key.weight | 6,144 | 0.0104 | 3.428 | 35.116 |
| bert.encoder.layer.10.attention.self.query.weight | 6,144 | 0.0104 | 3.514 | 35.312 |
| bert.encoder.layer.10.attention.self.value.weight | 6,144 | 0.0104 | 2.984 | 27.332 |
| bert.encoder.layer.10.intermediate.dense.weight | 24,576 | 0.0104 | 5.929 | 59.818 |
| bert.encoder.layer.10.output.LayerNorm.bias | 8 | 0.0104 | 0.098 | 2.194 |
| bert.encoder.layer.10.output.LayerNorm.weight | 8 | 0.0104 | 0.454 | 22.714 |
| bert.encoder.layer.10.output.dense.bias | 8 | 0.0104 | 0.245 | 2.841 |
| bert.encoder.layer.10.output.dense.weight | 24,576 | 0.0104 | 5.727 | 58.249 |
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.154 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.370 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 1.046 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 27712.797 | 28.241 |
| bert.encoder.layer.11.attention.self.query.bias | 768 | 1.0000 | 6.072 | 6.072 |
| bert.encoder.layer.11.attention.self.query.weight | 589,824 | 1.0000 | 34.948 | 34.948 |
| bert.encoder.layer.11.attention.self.value.bias | 768 | 1.0000 | 0.561 | 0.561 |
| bert.encoder.layer.11.attention.self.value.weight | 589,824 | 1.0000 | 40.898 | 30.114 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 1.471 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 10.198 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 1.590 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,296 | 1.0000 | 55.033 | 55.033 |
| bert.encoder.layer.2.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.181 | 4.892 |
| bert.encoder.layer.2.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.304 | 24.129 |
| bert.encoder.layer.2.attention.output.dense.bias | 8 | 0.0104 | 0.129 | 2.478 |
| bert.encoder.layer.2.attention.output.dense.weight | 6,144 | 0.0104 | 1.790 | 20.806 |
| bert.encoder.layer.2.attention.self.key.weight | 6,144 | 0.0104 | 3.330 | 35.837 |
| bert.encoder.layer.2.attention.self.query.weight | 6,144 | 0.0104 | 3.388 | 36.668 |
| bert.encoder.layer.2.attention.self.value.weight | 6,144 | 0.0104 | 2.092 | 21.429 |
| bert.encoder.layer.2.intermediate.dense.weight | 24,576 | 0.0104 | 5.818 | 61.253 |
| bert.encoder.layer.2.output.LayerNorm.bias | 8 | 0.0104 | 0.169 | 2.243 |
| bert.encoder.layer.2.output.LayerNorm.weight | 8 | 0.0104 | 0.316 | 23.637 |
| bert.encoder.layer.2.output.dense.bias | 8 | 0.0104 | 0.117 | 1.982 |
| bert.encoder.layer.2.output.dense.weight | 24,576 | 0.0104 | 5.450 | 58.770 |
| bert.encoder.layer.3.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.101 | 4.432 |
| bert.encoder.layer.3.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.335 | 24.131 |
| bert.encoder.layer.3.attention.output.dense.bias | 8 | 0.0104 | 0.186 | 1.995 |
| bert.encoder.layer.3.attention.output.dense.weight | 6,144 | 0.0104 | 1.913 | 22.318 |
| bert.encoder.layer.3.attention.self.key.weight | 6,144 | 0.0104 | 3.211 | 32.894 |
| bert.encoder.layer.3.attention.self.query.weight | 6,144 | 0.0104 | 3.219 | 33.031 |
| bert.encoder.layer.3.attention.self.value.weight | 6,144 | 0.0104 | 2.308 | 23.935 |
| bert.encoder.layer.3.intermediate.dense.weight | 24,576 | 0.0104 | 5.892 | 62.134 |
| bert.encoder.layer.3.output.LayerNorm.bias | 8 | 0.0104 | 0.126 | 1.716 |
| bert.encoder.layer.3.output.LayerNorm.weight | 8 | 0.0104 | 0.437 | 22.510 |
| bert.encoder.layer.3.output.dense.bias | 8 | 0.0104 | 0.094 | 2.028 |
| bert.encoder.layer.3.output.dense.weight | 24,576 | 0.0104 | 5.483 | 60.373 |
| bert.encoder.layer.4.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.133 | 4.025 |
| bert.encoder.layer.4.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.403 | 23.493 |
| bert.encoder.layer.4.attention.output.dense.bias | 8 | 0.0104 | 0.071 | 1.151 |
| bert.encoder.layer.4.attention.output.dense.weight | 6,144 | 0.0104 | 2.125 | 24.973 |
| bert.encoder.layer.4.attention.self.key.weight | 6,144 | 0.0104 | 3.179 | 32.130 |
| bert.encoder.layer.4.attention.self.query.weight | 6,144 | 0.0104 | 3.151 | 32.318 |
| bert.encoder.layer.4.attention.self.value.weight | 6,144 | 0.0104 | 2.525 | 26.809 |
| bert.encoder.layer.4.intermediate.dense.weight | 24,576 | 0.0104 | 5.954 | 62.600 |
| bert.encoder.layer.4.output.LayerNorm.bias | 8 | 0.0104 | 0.092 | 1.651 |
| bert.encoder.layer.4.output.LayerNorm.weight | 8 | 0.0104 | 0.361 | 23.304 |
| bert.encoder.layer.4.output.dense.bias | 8 | 0.0104 | 0.122 | 1.495 |
| bert.encoder.layer.4.output.dense.weight | 24,576 | 0.0104 | 5.578 | 61.223 |
| bert.encoder.layer.5.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.132 | 3.815 |
| bert.encoder.layer.5.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.364 | 23.662 |
| bert.encoder.layer.5.attention.output.dense.bias | 8 | 0.0104 | 0.056 | 1.210 |
| bert.encoder.layer.5.attention.output.dense.weight | 6,144 | 0.0104 | 2.188 | 25.461 |
| bert.encoder.layer.5.attention.self.key.weight | 6,144 | 0.0104 | 3.250 | 33.288 |
| bert.encoder.layer.5.attention.self.query.weight | 6,144 | 0.0104 | 3.296 | 33.310 |
| bert.encoder.layer.5.attention.self.value.weight | 6,144 | 0.0104 | 2.526 | 26.821 |
| bert.encoder.layer.5.intermediate.dense.weight | 24,576 | 0.0104 | 5.966 | 62.279 |
| bert.encoder.layer.5.output.LayerNorm.bias | 8 | 0.0104 | 0.113 | 1.872 |
| bert.encoder.layer.5.output.LayerNorm.weight | 8 | 0.0104 | 0.382 | 23.086 |
| bert.encoder.layer.5.output.dense.bias | 8 | 0.0104 | 0.090 | 1.601 |
| bert.encoder.layer.5.output.dense.weight | 24,576 | 0.0104 | 5.549 | 60.355 |
| bert.encoder.layer.6.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.180 | 3.942 |
| bert.encoder.layer.6.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.419 | 23.128 |
| bert.encoder.layer.6.attention.output.dense.bias | 8 | 0.0104 | 0.107 | 1.399 |
| bert.encoder.layer.6.attention.output.dense.weight | 6,144 | 0.0104 | 2.179 | 24.746 |
| bert.encoder.layer.6.attention.self.key.weight | 6,144 | 0.0104 | 3.182 | 32.990 |
| bert.encoder.layer.6.attention.self.query.weight | 6,144 | 0.0104 | 3.197 | 33.008 |
| bert.encoder.layer.6.attention.self.value.weight | 6,144 | 0.0104 | 2.541 | 26.145 |
| bert.encoder.layer.6.intermediate.dense.weight | 24,576 | 0.0104 | 6.057 | 62.594 |
| bert.encoder.layer.6.output.LayerNorm.bias | 8 | 0.0104 | 0.101 | 1.983 |
| bert.encoder.layer.6.output.LayerNorm.weight | 8 | 0.0104 | 0.407 | 23.124 |
| bert.encoder.layer.6.output.dense.bias | 8 | 0.0104 | 0.141 | 1.945 |
| bert.encoder.layer.6.output.dense.weight | 24,576 | 0.0104 | 5.523 | 59.579 |
| bert.encoder.layer.7.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.333 | 4.049 |
| bert.encoder.layer.7.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.434 | 22.970 |
| bert.encoder.layer.7.attention.output.dense.bias | 8 | 0.0104 | 0.124 | 1.738 |
| bert.encoder.layer.7.attention.output.dense.weight | 6,144 | 0.0104 | 2.165 | 24.337 |
| bert.encoder.layer.7.attention.self.key.weight | 6,144 | 0.0104 | 3.249 | 33.400 |
| bert.encoder.layer.7.attention.self.query.weight | 6,144 | 0.0104 | 3.177 | 33.299 |
| bert.encoder.layer.7.attention.self.value.weight | 6,144 | 0.0104 | 2.485 | 25.273 |
| bert.encoder.layer.7.intermediate.dense.weight | 24,576 | 0.0104 | 5.898 | 60.073 |
| bert.encoder.layer.7.output.LayerNorm.bias | 8 | 0.0104 | 0.081 | 2.017 |
| bert.encoder.layer.7.output.LayerNorm.weight | 8 | 0.0104 | 0.481 | 22.477 |
| bert.encoder.layer.7.output.dense.bias | 8 | 0.0104 | 0.164 | 2.260 |
| bert.encoder.layer.7.output.dense.weight | 24,576 | 0.0104 | 5.451 | 57.601 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.288 | 3.613 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.406 | 23.364 |
| bert.encoder.layer.8.attention.output.dense.bias | 8 | 0.0104 | 0.141 | 1.769 |
| bert.encoder.layer.8.attention.output.dense.weight | 6,144 | 0.0104 | 2.347 | 25.772 |
| bert.encoder.layer.8.attention.self.key.weight | 6,144 | 0.0104 | 3.246 | 33.893 |
| bert.encoder.layer.8.attention.self.query.weight | 6,144 | 0.0104 | 3.307 | 33.808 |
| bert.encoder.layer.8.attention.self.value.weight | 6,144 | 0.0104 | 2.735 | 27.192 |
| bert.encoder.layer.8.intermediate.dense.weight | 24,576 | 0.0104 | 5.973 | 59.920 |
| bert.encoder.layer.8.output.LayerNorm.bias | 8 | 0.0104 | 0.106 | 1.962 |
| bert.encoder.layer.8.output.LayerNorm.weight | 8 | 0.0104 | 0.463 | 23.060 |
| bert.encoder.layer.8.output.dense.bias | 8 | 0.0104 | 0.142 | 2.401 |
| bert.encoder.layer.8.output.dense.weight | 24,576 | 0.0104 | 5.520 | 57.365 |
| bert.encoder.layer.9.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.301 | 3.780 |
| bert.encoder.layer.9.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.491 | 22.757 |
| bert.encoder.layer.9.attention.output.dense.bias | 8 | 0.0104 | 0.155 | 1.703 |
| bert.encoder.layer.9.attention.output.dense.weight | 6,144 | 0.0104 | 2.381 | 25.113 |
| bert.encoder.layer.9.attention.self.key.weight | 6,144 | 0.0104 | 3.363 | 35.270 |
| bert.encoder.layer.9.attention.self.query.weight | 6,144 | 0.0104 | 3.508 | 35.247 |
| bert.encoder.layer.9.attention.self.value.weight | 6,144 | 0.0104 | 2.763 | 26.491 |
| bert.encoder.layer.9.intermediate.dense.weight | 24,576 | 0.0104 | 6.021 | 60.958 |
| bert.encoder.layer.9.output.LayerNorm.bias | 8 | 0.0104 | 0.094 | 1.811 |
| bert.encoder.layer.9.output.LayerNorm.weight | 8 | 0.0104 | 0.506 | 22.247 |
| bert.encoder.layer.9.output.dense.bias | 8 | 0.0104 | 0.140 | 2.219 |
| bert.encoder.layer.9.output.dense.weight | 24,576 | 0.0104 | 5.754 | 59.883 |
| bert.pooler.dense.bias | 768 | 1.0000 | 0.962 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 21.922 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 28.284 | 0.000 |

## Editor's final answer

I turned bert-base-uncased into an 8-label insurance classifier using closed-form maths only. Every weight was computed from frozen-model activations on practice sentences; nothing was trained iteratively and I wrote no text of my own.

**Expected accuracy: about 97.6–97.8%.** That comes from nested 5-fold cross-validation on the practice set. A final check on 512 practice sentences scored 99.8% (one error: a waiver-of-premium sentence labelled deductible_limits was predicted as premium_payment). That figure is inflated, though, because those sentences were also used to fit the edits, so it doesn't measure generalisation. I spent 9 of the 10 forward passes; the last was left unused because it couldn't provide an unbiased check.

**What the frozen model's activations showed:** a simple linear readout of the [CLS] vector works poorly. Averaging the hidden states over all tokens of the sentence works much better, about 96% from layers 8–11. Adding information from the first layer on top of that brings it to about 97.5%, because early layers carry word-level cues.

**What I changed:**
1. **Layer 0 (an early side channel):** I took over 8 low-impact neurons in the first feed-forward block and 8 low-variance hidden dimensions. Each token now writes an 8-dimensional projection of its early representation into those dimensions. The projection was fitted by ridge regression so that it adds what the later layers miss. The neurons are biased to stay in their near-linear range.
2. **Layers 1–10 (protecting the channel):** in every block I zeroed the weights that read from or write to those 8 dimensions, and set their LayerNorm scale to 1 and shift to 0. The early signal therefore passes through to the top unchanged.
3. **Layer 11 (mean pooling):** I set the attention query weights to zero, so the [CLS] token attends equally to every real token (padding stays masked). I made the value projection the identity and scaled the output projection up by 1000, so its result swamps the residual. I zeroed the feed-forward output and reset the LayerNorm scale to 1 and shift to 0. The final [CLS] vector is then essentially the normalised average of all token states. It carries both the deep features and the protected early channel, and works for any sentence length.
4. **Pooler and classifier (closed-form readout):** I read the edited model's final [CLS] vectors for all 1496 practice sentences and fitted a ridge-regression classifier on them (regularisation 0.2, chosen by nested cross-validation). The first 8 rows of the pooler hold 0.1 times this classifier, so the tanh stays nearly linear; all other rows are zero. The classifier weight is 10 times the identity on those 8 pooled dimensions, with zero bias. Because tanh only ever increases, the predicted label is exactly the ridge classifier's choice.

**Confirming the pipeline:** I simulated the pooler and classifier offline from the [CLS] vectors I had read out. On the 512-sentence check, the model's logits matched that simulation to within 5e-5, so the edits behave as designed end to end.

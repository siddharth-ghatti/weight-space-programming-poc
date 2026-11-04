# Evaluation: `<repo>/runs/_baselines/finetune/model`

**Test accuracy 0.9725 | macro-F1 0.9727 | weighted-F1 0.9724 | n=1779**
Practice accuracy 0.9940 | macro-F1 0.9940 | gap +0.0215

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.974 | 0.948 | 0.961 | 233 |
| exclusion | 0.970 | 0.987 | 0.978 | 226 |
| premium_payment | 0.964 | 0.977 | 0.971 | 219 |
| deductible_limits | 0.965 | 0.978 | 0.971 | 225 |
| claims_procedure | 0.969 | 0.982 | 0.975 | 222 |
| definitions | 1.000 | 0.995 | 0.998 | 206 |
| term_cancellation_renewal | 0.982 | 0.973 | 0.977 | 223 |
| conditions_duties | 0.959 | 0.942 | 0.951 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 221 | 4 | 0 | 3 | 0 | 0 | 1 | 4 |
| 1 exclusion | 1 | 223 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2 premium_payment | 0 | 0 | 214 | 5 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 2 | 2 | 1 | 220 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 0 | 0 | 0 | 218 | 0 | 0 | 3 |
| 5 definitions | 0 | 1 | 0 | 0 | 0 | 205 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 0 | 5 | 0 | 0 | 0 | 217 | 1 |
| 7 conditions_duties | 2 | 0 | 2 | 0 | 7 | 0 | 2 | 212 |

## Edited tensors

201 tensors, 109,488,076 entries changed (99.9997% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.embeddings.LayerNorm.bias | 768 | 1.0000 | 0.007 | 1.994 |
| bert.embeddings.LayerNorm.weight | 768 | 1.0000 | 0.008 | 23.827 |
| bert.embeddings.position_embeddings.weight | 393,215 | 1.0000 | 0.050 | 10.076 |
| bert.embeddings.token_type_embeddings.weight | 1,536 | 1.0000 | 0.007 | 1.315 |
| bert.embeddings.word_embeddings.weight | 23,440,877 | 1.0000 | 0.320 | 247.141 |
| bert.encoder.layer.0.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 9.131 |
| bert.encoder.layer.0.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 26.663 |
| bert.encoder.layer.0.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 0.983 |
| bert.encoder.layer.0.attention.output.dense.weight | 589,824 | 1.0000 | 0.221 | 21.947 |
| bert.encoder.layer.0.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.089 |
| bert.encoder.layer.0.attention.self.key.weight | 589,822 | 1.0000 | 0.198 | 32.862 |
| bert.encoder.layer.0.attention.self.query.bias | 768 | 1.0000 | 0.009 | 7.781 |
| bert.encoder.layer.0.attention.self.query.weight | 589,820 | 1.0000 | 0.197 | 33.116 |
| bert.encoder.layer.0.attention.self.value.bias | 768 | 1.0000 | 0.007 | 1.129 |
| bert.encoder.layer.0.attention.self.value.weight | 589,823 | 1.0000 | 0.218 | 22.127 |
| bert.encoder.layer.0.intermediate.dense.bias | 3,072 | 1.0000 | 0.015 | 6.588 |
| bert.encoder.layer.0.intermediate.dense.weight | 2,359,286 | 1.0000 | 0.410 | 57.479 |
| bert.encoder.layer.0.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 2.912 |
| bert.encoder.layer.0.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 21.005 |
| bert.encoder.layer.0.output.dense.bias | 768 | 1.0000 | 0.007 | 2.222 |
| bert.encoder.layer.0.output.dense.weight | 2,359,286 | 1.0000 | 0.409 | 55.062 |
| bert.encoder.layer.1.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 5.384 |
| bert.encoder.layer.1.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 24.390 |
| bert.encoder.layer.1.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 1.673 |
| bert.encoder.layer.1.attention.output.dense.weight | 589,818 | 1.0000 | 0.209 | 21.504 |
| bert.encoder.layer.1.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.142 |
| bert.encoder.layer.1.attention.self.key.weight | 589,823 | 1.0000 | 0.189 | 32.659 |
| bert.encoder.layer.1.attention.self.query.bias | 768 | 1.0000 | 0.007 | 5.024 |
| bert.encoder.layer.1.attention.self.query.weight | 589,820 | 1.0000 | 0.193 | 32.666 |
| bert.encoder.layer.1.attention.self.value.bias | 768 | 1.0000 | 0.007 | 1.161 |
| bert.encoder.layer.1.attention.self.value.weight | 589,823 | 1.0000 | 0.208 | 21.783 |
| bert.encoder.layer.1.intermediate.dense.bias | 3,072 | 1.0000 | 0.015 | 6.174 |
| bert.encoder.layer.1.intermediate.dense.weight | 2,359,285 | 1.0000 | 0.411 | 60.250 |
| bert.encoder.layer.1.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 2.564 |
| bert.encoder.layer.1.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 24.173 |
| bert.encoder.layer.1.output.dense.bias | 768 | 1.0000 | 0.007 | 1.679 |
| bert.encoder.layer.1.output.dense.weight | 2,359,286 | 1.0000 | 0.402 | 58.159 |
| bert.encoder.layer.10.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 3.648 |
| bert.encoder.layer.10.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.011 | 23.585 |
| bert.encoder.layer.10.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 1.460 |
| bert.encoder.layer.10.attention.output.dense.weight | 589,823 | 1.0000 | 0.326 | 25.615 |
| bert.encoder.layer.10.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.176 |
| bert.encoder.layer.10.attention.self.key.weight | 589,819 | 1.0000 | 0.209 | 35.116 |
| bert.encoder.layer.10.attention.self.query.bias | 768 | 1.0000 | 0.009 | 6.376 |
| bert.encoder.layer.10.attention.self.query.weight | 589,824 | 1.0000 | 0.214 | 35.312 |
| bert.encoder.layer.10.attention.self.value.bias | 768 | 1.0000 | 0.007 | 0.782 |
| bert.encoder.layer.10.attention.self.value.weight | 589,820 | 1.0000 | 0.334 | 27.332 |
| bert.encoder.layer.10.intermediate.dense.bias | 3,072 | 1.0000 | 0.024 | 6.158 |
| bert.encoder.layer.10.intermediate.dense.weight | 2,359,284 | 1.0000 | 0.637 | 59.818 |
| bert.encoder.layer.10.output.LayerNorm.bias | 768 | 1.0000 | 0.009 | 2.194 |
| bert.encoder.layer.10.output.LayerNorm.weight | 768 | 1.0000 | 0.013 | 22.714 |
| bert.encoder.layer.10.output.dense.bias | 768 | 1.0000 | 0.009 | 2.841 |
| bert.encoder.layer.10.output.dense.weight | 2,359,294 | 1.0000 | 0.589 | 58.249 |
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.011 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.017 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 0.010 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 0.453 | 28.241 |
| bert.encoder.layer.11.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.146 |
| bert.encoder.layer.11.attention.self.key.weight | 589,821 | 1.0000 | 0.255 | 34.171 |
| bert.encoder.layer.11.attention.self.query.bias | 768 | 1.0000 | 0.011 | 6.072 |
| bert.encoder.layer.11.attention.self.query.weight | 589,820 | 1.0000 | 0.269 | 34.948 |
| bert.encoder.layer.11.attention.self.value.bias | 768 | 1.0000 | 0.009 | 0.561 |
| bert.encoder.layer.11.attention.self.value.weight | 589,823 | 1.0000 | 0.454 | 30.114 |
| bert.encoder.layer.11.intermediate.dense.bias | 3,072 | 1.0000 | 0.036 | 5.159 |
| bert.encoder.layer.11.intermediate.dense.weight | 2,359,291 | 1.0000 | 0.948 | 60.183 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 0.011 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 0.037 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 0.013 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,295 | 1.0000 | 0.962 | 55.033 |
| bert.encoder.layer.2.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 4.892 |
| bert.encoder.layer.2.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 24.129 |
| bert.encoder.layer.2.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 2.478 |
| bert.encoder.layer.2.attention.output.dense.weight | 589,824 | 1.0000 | 0.208 | 20.806 |
| bert.encoder.layer.2.attention.self.key.bias | 766 | 0.9974 | 0.000 | 0.116 |
| bert.encoder.layer.2.attention.self.key.weight | 589,820 | 1.0000 | 0.193 | 35.837 |
| bert.encoder.layer.2.attention.self.query.bias | 768 | 1.0000 | 0.007 | 3.590 |
| bert.encoder.layer.2.attention.self.query.weight | 589,824 | 1.0000 | 0.194 | 36.668 |
| bert.encoder.layer.2.attention.self.value.bias | 768 | 1.0000 | 0.007 | 1.598 |
| bert.encoder.layer.2.attention.self.value.weight | 589,823 | 1.0000 | 0.206 | 21.429 |
| bert.encoder.layer.2.intermediate.dense.bias | 3,072 | 1.0000 | 0.015 | 6.411 |
| bert.encoder.layer.2.intermediate.dense.weight | 2,359,286 | 1.0000 | 0.411 | 61.253 |
| bert.encoder.layer.2.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 2.243 |
| bert.encoder.layer.2.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 23.637 |
| bert.encoder.layer.2.output.dense.bias | 768 | 1.0000 | 0.007 | 1.982 |
| bert.encoder.layer.2.output.dense.weight | 2,359,283 | 1.0000 | 0.404 | 58.770 |
| bert.encoder.layer.3.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 4.432 |
| bert.encoder.layer.3.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 24.131 |
| bert.encoder.layer.3.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 1.995 |
| bert.encoder.layer.3.attention.output.dense.weight | 589,822 | 1.0000 | 0.211 | 22.318 |
| bert.encoder.layer.3.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.133 |
| bert.encoder.layer.3.attention.self.key.weight | 589,824 | 1.0000 | 0.199 | 32.894 |
| bert.encoder.layer.3.attention.self.query.bias | 768 | 1.0000 | 0.007 | 3.819 |
| bert.encoder.layer.3.attention.self.query.weight | 589,822 | 1.0000 | 0.198 | 33.031 |
| bert.encoder.layer.3.attention.self.value.bias | 768 | 1.0000 | 0.007 | 0.907 |
| bert.encoder.layer.3.attention.self.value.weight | 589,822 | 1.0000 | 0.208 | 23.935 |
| bert.encoder.layer.3.intermediate.dense.bias | 3,072 | 1.0000 | 0.015 | 6.232 |
| bert.encoder.layer.3.intermediate.dense.weight | 2,359,289 | 1.0000 | 0.416 | 62.134 |
| bert.encoder.layer.3.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 1.716 |
| bert.encoder.layer.3.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 22.510 |
| bert.encoder.layer.3.output.dense.bias | 768 | 1.0000 | 0.007 | 2.028 |
| bert.encoder.layer.3.output.dense.weight | 2,359,290 | 1.0000 | 0.405 | 60.373 |
| bert.encoder.layer.4.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 4.025 |
| bert.encoder.layer.4.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 23.493 |
| bert.encoder.layer.4.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 1.151 |
| bert.encoder.layer.4.attention.output.dense.weight | 589,822 | 1.0000 | 0.222 | 24.973 |
| bert.encoder.layer.4.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.134 |
| bert.encoder.layer.4.attention.self.key.weight | 589,824 | 1.0000 | 0.204 | 32.130 |
| bert.encoder.layer.4.attention.self.query.bias | 768 | 1.0000 | 0.008 | 4.094 |
| bert.encoder.layer.4.attention.self.query.weight | 589,820 | 1.0000 | 0.203 | 32.318 |
| bert.encoder.layer.4.attention.self.value.bias | 768 | 1.0000 | 0.007 | 0.743 |
| bert.encoder.layer.4.attention.self.value.weight | 589,823 | 1.0000 | 0.218 | 26.809 |
| bert.encoder.layer.4.intermediate.dense.bias | 3,072 | 1.0000 | 0.016 | 6.676 |
| bert.encoder.layer.4.intermediate.dense.weight | 2,359,285 | 1.0000 | 0.420 | 62.600 |
| bert.encoder.layer.4.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 1.651 |
| bert.encoder.layer.4.output.LayerNorm.weight | 768 | 1.0000 | 0.008 | 23.304 |
| bert.encoder.layer.4.output.dense.bias | 768 | 1.0000 | 0.007 | 1.495 |
| bert.encoder.layer.4.output.dense.weight | 2,359,286 | 1.0000 | 0.415 | 61.223 |
| bert.encoder.layer.5.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.007 | 3.815 |
| bert.encoder.layer.5.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 23.662 |
| bert.encoder.layer.5.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 1.210 |
| bert.encoder.layer.5.attention.output.dense.weight | 589,822 | 1.0000 | 0.234 | 25.461 |
| bert.encoder.layer.5.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.166 |
| bert.encoder.layer.5.attention.self.key.weight | 589,823 | 1.0000 | 0.201 | 33.288 |
| bert.encoder.layer.5.attention.self.query.bias | 768 | 1.0000 | 0.008 | 3.492 |
| bert.encoder.layer.5.attention.self.query.weight | 589,820 | 1.0000 | 0.202 | 33.310 |
| bert.encoder.layer.5.attention.self.value.bias | 768 | 1.0000 | 0.007 | 0.759 |
| bert.encoder.layer.5.attention.self.value.weight | 589,823 | 1.0000 | 0.225 | 26.821 |
| bert.encoder.layer.5.intermediate.dense.bias | 3,072 | 1.0000 | 0.017 | 6.765 |
| bert.encoder.layer.5.intermediate.dense.weight | 2,359,283 | 1.0000 | 0.429 | 62.279 |
| bert.encoder.layer.5.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 1.872 |
| bert.encoder.layer.5.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 23.086 |
| bert.encoder.layer.5.output.dense.bias | 768 | 1.0000 | 0.007 | 1.601 |
| bert.encoder.layer.5.output.dense.weight | 2,359,282 | 1.0000 | 0.421 | 60.355 |
| bert.encoder.layer.6.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 3.942 |
| bert.encoder.layer.6.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 23.128 |
| bert.encoder.layer.6.attention.output.dense.bias | 768 | 1.0000 | 0.008 | 1.399 |
| bert.encoder.layer.6.attention.output.dense.weight | 589,819 | 1.0000 | 0.237 | 24.746 |
| bert.encoder.layer.6.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.148 |
| bert.encoder.layer.6.attention.self.key.weight | 589,821 | 1.0000 | 0.210 | 32.990 |
| bert.encoder.layer.6.attention.self.query.bias | 768 | 1.0000 | 0.009 | 3.551 |
| bert.encoder.layer.6.attention.self.query.weight | 589,823 | 1.0000 | 0.210 | 33.008 |
| bert.encoder.layer.6.attention.self.value.bias | 768 | 1.0000 | 0.008 | 0.978 |
| bert.encoder.layer.6.attention.self.value.weight | 589,821 | 1.0000 | 0.235 | 26.145 |
| bert.encoder.layer.6.intermediate.dense.bias | 3,072 | 1.0000 | 0.016 | 6.646 |
| bert.encoder.layer.6.intermediate.dense.weight | 2,359,292 | 1.0000 | 0.431 | 62.594 |
| bert.encoder.layer.6.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 1.983 |
| bert.encoder.layer.6.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 23.124 |
| bert.encoder.layer.6.output.dense.bias | 768 | 1.0000 | 0.007 | 1.945 |
| bert.encoder.layer.6.output.dense.weight | 2,359,291 | 1.0000 | 0.426 | 59.579 |
| bert.encoder.layer.7.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 4.049 |
| bert.encoder.layer.7.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 22.970 |
| bert.encoder.layer.7.attention.output.dense.bias | 768 | 1.0000 | 0.008 | 1.738 |
| bert.encoder.layer.7.attention.output.dense.weight | 589,823 | 1.0000 | 0.246 | 24.337 |
| bert.encoder.layer.7.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.150 |
| bert.encoder.layer.7.attention.self.key.weight | 589,823 | 1.0000 | 0.206 | 33.400 |
| bert.encoder.layer.7.attention.self.query.bias | 768 | 1.0000 | 0.008 | 4.020 |
| bert.encoder.layer.7.attention.self.query.weight | 589,821 | 1.0000 | 0.202 | 33.299 |
| bert.encoder.layer.7.attention.self.value.bias | 768 | 1.0000 | 0.008 | 1.147 |
| bert.encoder.layer.7.attention.self.value.weight | 589,824 | 1.0000 | 0.241 | 25.273 |
| bert.encoder.layer.7.intermediate.dense.bias | 3,072 | 1.0000 | 0.017 | 6.633 |
| bert.encoder.layer.7.intermediate.dense.weight | 2,359,286 | 1.0000 | 0.446 | 60.073 |
| bert.encoder.layer.7.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 2.017 |
| bert.encoder.layer.7.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 22.477 |
| bert.encoder.layer.7.output.dense.bias | 768 | 1.0000 | 0.008 | 2.260 |
| bert.encoder.layer.7.output.dense.weight | 2,359,292 | 1.0000 | 0.446 | 57.601 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 3.613 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.010 | 23.364 |
| bert.encoder.layer.8.attention.output.dense.bias | 768 | 1.0000 | 0.008 | 1.769 |
| bert.encoder.layer.8.attention.output.dense.weight | 589,823 | 1.0000 | 0.279 | 25.772 |
| bert.encoder.layer.8.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.184 |
| bert.encoder.layer.8.attention.self.key.weight | 589,823 | 1.0000 | 0.220 | 33.893 |
| bert.encoder.layer.8.attention.self.query.bias | 768 | 1.0000 | 0.010 | 4.838 |
| bert.encoder.layer.8.attention.self.query.weight | 589,822 | 1.0000 | 0.224 | 33.808 |
| bert.encoder.layer.8.attention.self.value.bias | 768 | 1.0000 | 0.009 | 1.227 |
| bert.encoder.layer.8.attention.self.value.weight | 589,824 | 1.0000 | 0.269 | 27.192 |
| bert.encoder.layer.8.intermediate.dense.bias | 3,072 | 1.0000 | 0.018 | 6.144 |
| bert.encoder.layer.8.intermediate.dense.weight | 2,359,289 | 1.0000 | 0.465 | 59.920 |
| bert.encoder.layer.8.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 1.962 |
| bert.encoder.layer.8.output.LayerNorm.weight | 768 | 1.0000 | 0.009 | 23.060 |
| bert.encoder.layer.8.output.dense.bias | 768 | 1.0000 | 0.007 | 2.401 |
| bert.encoder.layer.8.output.dense.weight | 2,359,287 | 1.0000 | 0.449 | 57.365 |
| bert.encoder.layer.9.attention.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 3.780 |
| bert.encoder.layer.9.attention.output.LayerNorm.weight | 768 | 1.0000 | 0.010 | 22.757 |
| bert.encoder.layer.9.attention.output.dense.bias | 768 | 1.0000 | 0.007 | 1.703 |
| bert.encoder.layer.9.attention.output.dense.weight | 589,820 | 1.0000 | 0.283 | 25.113 |
| bert.encoder.layer.9.attention.self.key.bias | 768 | 1.0000 | 0.000 | 0.184 |
| bert.encoder.layer.9.attention.self.key.weight | 589,823 | 1.0000 | 0.222 | 35.270 |
| bert.encoder.layer.9.attention.self.query.bias | 768 | 1.0000 | 0.010 | 5.314 |
| bert.encoder.layer.9.attention.self.query.weight | 589,821 | 1.0000 | 0.223 | 35.247 |
| bert.encoder.layer.9.attention.self.value.bias | 768 | 1.0000 | 0.008 | 1.053 |
| bert.encoder.layer.9.attention.self.value.weight | 589,822 | 1.0000 | 0.283 | 26.491 |
| bert.encoder.layer.9.intermediate.dense.bias | 3,072 | 1.0000 | 0.019 | 6.363 |
| bert.encoder.layer.9.intermediate.dense.weight | 2,359,289 | 1.0000 | 0.510 | 60.958 |
| bert.encoder.layer.9.output.LayerNorm.bias | 768 | 1.0000 | 0.008 | 1.811 |
| bert.encoder.layer.9.output.LayerNorm.weight | 768 | 1.0000 | 0.010 | 22.247 |
| bert.encoder.layer.9.output.dense.bias | 768 | 1.0000 | 0.008 | 2.219 |
| bert.encoder.layer.9.output.dense.weight | 2,359,289 | 1.0000 | 0.483 | 59.883 |
| bert.pooler.dense.bias | 768 | 1.0000 | 0.015 | 0.961 |
| bert.pooler.dense.weight | 589,822 | 1.0000 | 0.798 | 21.921 |
| classifier.bias | 8 | 1.0000 | 0.001 | 0.000 |
| classifier.weight | 6,144 | 1.0000 | 1.576 | 0.000 |

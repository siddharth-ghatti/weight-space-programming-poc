# Evaluation: `<repo>/runs/v1/p25/model`

**Test accuracy 0.9820 | macro-F1 0.9822 | weighted-F1 0.9820 | n=1779**
Practice accuracy 0.9993 | macro-F1 0.9993 | gap +0.0173

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.974 | 0.970 | 0.972 | 233 |
| exclusion | 0.982 | 0.978 | 0.980 | 226 |
| premium_payment | 0.964 | 0.991 | 0.977 | 219 |
| deductible_limits | 0.991 | 0.987 | 0.989 | 225 |
| claims_procedure | 0.982 | 0.995 | 0.989 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 0.982 | 0.969 | 0.975 | 223 |
| conditions_duties | 0.991 | 0.969 | 0.980 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 226 | 3 | 0 | 1 | 0 | 0 | 2 | 1 |
| 1 exclusion | 1 | 221 | 0 | 0 | 0 | 2 | 1 | 1 |
| 2 premium_payment | 1 | 0 | 217 | 1 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 1 | 1 | 1 | 222 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 0 | 0 | 0 | 221 | 0 | 0 | 0 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 6 | 0 | 0 | 0 | 216 | 0 |
| 7 conditions_duties | 1 | 0 | 1 | 0 | 4 | 0 | 1 | 218 |

## Edited tensors

176 tensors, 11,386,808 entries changed (10.4000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.0.attention.output.LayerNorm.bias | 8 | 0.0104 | 3.265 | 9.131 |
| bert.encoder.layer.0.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.810 | 26.663 |
| bert.encoder.layer.0.attention.output.dense.bias | 768 | 1.0000 | 0.355 | 0.983 |
| bert.encoder.layer.0.attention.output.dense.weight | 12,224 | 0.0207 | 12.375 | 21.947 |
| bert.encoder.layer.0.attention.self.key.bias | 0 | 0.0000 | 0.000 | 0.089 |
| bert.encoder.layer.0.attention.self.key.weight | 0 | 0.0000 | 0.000 | 32.862 |
| bert.encoder.layer.0.attention.self.query.bias | 64 | 0.0833 | 2.514 | 7.781 |
| bert.encoder.layer.0.attention.self.query.weight | 49,152 | 0.0833 | 8.304 | 33.116 |
| bert.encoder.layer.0.attention.self.value.bias | 8 | 0.0104 | 0.274 | 1.129 |
| bert.encoder.layer.0.attention.self.value.weight | 6,144 | 0.0104 | 2.762 | 22.127 |
| bert.encoder.layer.0.intermediate.dense.bias | 3,072 | 1.0000 | 1.552 | 6.588 |
| bert.encoder.layer.0.intermediate.dense.weight | 24,576 | 0.0104 | 6.458 | 57.479 |
| bert.encoder.layer.0.output.LayerNorm.bias | 8 | 0.0104 | 0.665 | 2.912 |
| bert.encoder.layer.0.output.LayerNorm.weight | 8 | 0.0104 | 1.042 | 21.005 |
| bert.encoder.layer.0.output.dense.bias | 8 | 0.0104 | 0.387 | 2.222 |
| bert.encoder.layer.0.output.dense.weight | 24,576 | 0.0104 | 8.562 | 55.062 |
| bert.encoder.layer.1.attention.output.LayerNorm.bias | 8 | 0.0104 | 1.416 | 5.384 |
| bert.encoder.layer.1.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.437 | 24.390 |
| bert.encoder.layer.1.attention.output.dense.bias | 8 | 0.0104 | 0.253 | 1.673 |
| bert.encoder.layer.1.attention.output.dense.weight | 6,144 | 0.0104 | 3.929 | 21.504 |
| bert.encoder.layer.1.attention.self.key.bias | 768 | 1.0000 | 1.312 | 0.142 |
| bert.encoder.layer.1.attention.self.key.weight | 6,144 | 0.0104 | 4.988 | 32.659 |
| bert.encoder.layer.1.attention.self.query.bias | 768 | 1.0000 | 1.218 | 5.024 |
| bert.encoder.layer.1.attention.self.query.weight | 6,144 | 0.0104 | 4.645 | 32.666 |
| bert.encoder.layer.1.attention.self.value.bias | 768 | 1.0000 | 0.516 | 1.161 |
| bert.encoder.layer.1.attention.self.value.weight | 6,144 | 0.0104 | 2.020 | 21.783 |
| bert.encoder.layer.1.intermediate.dense.bias | 3,072 | 1.0000 | 1.816 | 6.174 |
| bert.encoder.layer.1.intermediate.dense.weight | 24,576 | 0.0104 | 7.277 | 60.250 |
| bert.encoder.layer.1.output.LayerNorm.bias | 8 | 0.0104 | 0.374 | 2.564 |
| bert.encoder.layer.1.output.LayerNorm.weight | 8 | 0.0104 | 0.912 | 24.173 |
| bert.encoder.layer.1.output.dense.bias | 8 | 0.0104 | 0.202 | 1.679 |
| bert.encoder.layer.1.output.dense.weight | 24,576 | 0.0104 | 9.489 | 58.159 |
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
| bert.encoder.layer.2.attention.output.LayerNorm.bias | 8 | 0.0104 | 1.069 | 4.892 |
| bert.encoder.layer.2.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.493 | 24.129 |
| bert.encoder.layer.2.attention.output.dense.bias | 8 | 0.0104 | 0.419 | 2.478 |
| bert.encoder.layer.2.attention.output.dense.weight | 6,144 | 0.0104 | 3.306 | 20.806 |
| bert.encoder.layer.2.attention.self.key.bias | 768 | 1.0000 | 1.045 | 0.116 |
| bert.encoder.layer.2.attention.self.key.weight | 6,144 | 0.0104 | 4.321 | 35.837 |
| bert.encoder.layer.2.attention.self.query.bias | 768 | 1.0000 | 0.961 | 3.590 |
| bert.encoder.layer.2.attention.self.query.weight | 6,144 | 0.0104 | 3.978 | 36.668 |
| bert.encoder.layer.2.attention.self.value.bias | 768 | 1.0000 | 0.615 | 1.598 |
| bert.encoder.layer.2.attention.self.value.weight | 6,144 | 0.0104 | 2.611 | 21.429 |
| bert.encoder.layer.2.intermediate.dense.bias | 3,072 | 1.0000 | 1.300 | 6.411 |
| bert.encoder.layer.2.intermediate.dense.weight | 24,576 | 0.0104 | 7.313 | 61.253 |
| bert.encoder.layer.2.output.LayerNorm.bias | 8 | 0.0104 | 0.308 | 2.243 |
| bert.encoder.layer.2.output.LayerNorm.weight | 8 | 0.0104 | 0.611 | 23.637 |
| bert.encoder.layer.2.output.dense.bias | 8 | 0.0104 | 0.274 | 1.982 |
| bert.encoder.layer.2.output.dense.weight | 24,576 | 0.0104 | 7.878 | 58.770 |
| bert.encoder.layer.3.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.867 | 4.432 |
| bert.encoder.layer.3.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.494 | 24.131 |
| bert.encoder.layer.3.attention.output.dense.bias | 8 | 0.0104 | 0.351 | 1.995 |
| bert.encoder.layer.3.attention.output.dense.weight | 6,144 | 0.0104 | 3.437 | 22.318 |
| bert.encoder.layer.3.attention.self.key.bias | 768 | 1.0000 | 0.656 | 0.133 |
| bert.encoder.layer.3.attention.self.key.weight | 6,144 | 0.0104 | 4.032 | 32.894 |
| bert.encoder.layer.3.attention.self.query.bias | 768 | 1.0000 | 0.675 | 3.819 |
| bert.encoder.layer.3.attention.self.query.weight | 6,144 | 0.0104 | 3.919 | 33.031 |
| bert.encoder.layer.3.attention.self.value.bias | 768 | 1.0000 | 0.546 | 0.907 |
| bert.encoder.layer.3.attention.self.value.weight | 6,144 | 0.0104 | 2.961 | 23.935 |
| bert.encoder.layer.3.intermediate.dense.bias | 3,072 | 1.0000 | 1.583 | 6.232 |
| bert.encoder.layer.3.intermediate.dense.weight | 24,576 | 0.0104 | 7.285 | 62.134 |
| bert.encoder.layer.3.output.LayerNorm.bias | 8 | 0.0104 | 0.207 | 1.716 |
| bert.encoder.layer.3.output.LayerNorm.weight | 8 | 0.0104 | 0.392 | 22.510 |
| bert.encoder.layer.3.output.dense.bias | 8 | 0.0104 | 0.308 | 2.028 |
| bert.encoder.layer.3.output.dense.weight | 24,576 | 0.0104 | 7.601 | 60.373 |
| bert.encoder.layer.4.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.690 | 4.025 |
| bert.encoder.layer.4.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.429 | 23.493 |
| bert.encoder.layer.4.attention.output.dense.bias | 8 | 0.0104 | 0.167 | 1.151 |
| bert.encoder.layer.4.attention.output.dense.weight | 6,144 | 0.0104 | 3.703 | 24.973 |
| bert.encoder.layer.4.attention.self.key.bias | 768 | 1.0000 | 0.751 | 0.134 |
| bert.encoder.layer.4.attention.self.key.weight | 6,144 | 0.0104 | 3.779 | 32.130 |
| bert.encoder.layer.4.attention.self.query.bias | 768 | 1.0000 | 0.749 | 4.094 |
| bert.encoder.layer.4.attention.self.query.weight | 6,144 | 0.0104 | 3.774 | 32.318 |
| bert.encoder.layer.4.attention.self.value.bias | 768 | 1.0000 | 0.695 | 0.743 |
| bert.encoder.layer.4.attention.self.value.weight | 6,144 | 0.0104 | 3.351 | 26.809 |
| bert.encoder.layer.4.intermediate.dense.bias | 3,072 | 1.0000 | 1.460 | 6.676 |
| bert.encoder.layer.4.intermediate.dense.weight | 24,576 | 0.0104 | 7.385 | 62.600 |
| bert.encoder.layer.4.output.LayerNorm.bias | 8 | 0.0104 | 0.200 | 1.651 |
| bert.encoder.layer.4.output.LayerNorm.weight | 8 | 0.0104 | 0.346 | 23.304 |
| bert.encoder.layer.4.output.dense.bias | 8 | 0.0104 | 0.227 | 1.495 |
| bert.encoder.layer.4.output.dense.weight | 24,576 | 0.0104 | 7.810 | 61.223 |
| bert.encoder.layer.5.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.681 | 3.815 |
| bert.encoder.layer.5.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.407 | 23.662 |
| bert.encoder.layer.5.attention.output.dense.bias | 8 | 0.0104 | 0.170 | 1.210 |
| bert.encoder.layer.5.attention.output.dense.weight | 6,144 | 0.0104 | 3.568 | 25.461 |
| bert.encoder.layer.5.attention.self.key.bias | 768 | 1.0000 | 0.710 | 0.166 |
| bert.encoder.layer.5.attention.self.key.weight | 6,144 | 0.0104 | 3.640 | 33.288 |
| bert.encoder.layer.5.attention.self.query.bias | 768 | 1.0000 | 0.721 | 3.492 |
| bert.encoder.layer.5.attention.self.query.weight | 6,144 | 0.0104 | 3.624 | 33.310 |
| bert.encoder.layer.5.attention.self.value.bias | 768 | 1.0000 | 0.657 | 0.759 |
| bert.encoder.layer.5.attention.self.value.weight | 6,144 | 0.0104 | 3.358 | 26.821 |
| bert.encoder.layer.5.intermediate.dense.bias | 3,072 | 1.0000 | 1.263 | 6.765 |
| bert.encoder.layer.5.intermediate.dense.weight | 24,576 | 0.0104 | 7.289 | 62.279 |
| bert.encoder.layer.5.output.LayerNorm.bias | 8 | 0.0104 | 0.206 | 1.872 |
| bert.encoder.layer.5.output.LayerNorm.weight | 8 | 0.0104 | 0.282 | 23.086 |
| bert.encoder.layer.5.output.dense.bias | 8 | 0.0104 | 0.220 | 1.601 |
| bert.encoder.layer.5.output.dense.weight | 24,576 | 0.0104 | 7.602 | 60.355 |
| bert.encoder.layer.6.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.659 | 3.942 |
| bert.encoder.layer.6.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.363 | 23.128 |
| bert.encoder.layer.6.attention.output.dense.bias | 8 | 0.0104 | 0.165 | 1.399 |
| bert.encoder.layer.6.attention.output.dense.weight | 6,144 | 0.0104 | 3.358 | 24.746 |
| bert.encoder.layer.6.attention.self.key.bias | 768 | 1.0000 | 0.608 | 0.148 |
| bert.encoder.layer.6.attention.self.key.weight | 6,144 | 0.0104 | 3.762 | 32.990 |
| bert.encoder.layer.6.attention.self.query.bias | 768 | 1.0000 | 0.595 | 3.551 |
| bert.encoder.layer.6.attention.self.query.weight | 6,144 | 0.0104 | 3.654 | 33.008 |
| bert.encoder.layer.6.attention.self.value.bias | 768 | 1.0000 | 0.463 | 0.978 |
| bert.encoder.layer.6.attention.self.value.weight | 6,144 | 0.0104 | 3.046 | 26.145 |
| bert.encoder.layer.6.intermediate.dense.bias | 3,072 | 1.0000 | 1.330 | 6.646 |
| bert.encoder.layer.6.intermediate.dense.weight | 24,576 | 0.0104 | 7.209 | 62.594 |
| bert.encoder.layer.6.output.LayerNorm.bias | 8 | 0.0104 | 0.217 | 1.983 |
| bert.encoder.layer.6.output.LayerNorm.weight | 8 | 0.0104 | 0.240 | 23.124 |
| bert.encoder.layer.6.output.dense.bias | 8 | 0.0104 | 0.232 | 1.945 |
| bert.encoder.layer.6.output.dense.weight | 24,576 | 0.0104 | 7.387 | 59.579 |
| bert.encoder.layer.7.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.548 | 4.049 |
| bert.encoder.layer.7.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.321 | 22.970 |
| bert.encoder.layer.7.attention.output.dense.bias | 8 | 0.0104 | 0.103 | 1.738 |
| bert.encoder.layer.7.attention.output.dense.weight | 6,144 | 0.0104 | 3.215 | 24.337 |
| bert.encoder.layer.7.attention.self.key.bias | 768 | 1.0000 | 0.636 | 0.150 |
| bert.encoder.layer.7.attention.self.key.weight | 6,144 | 0.0104 | 3.709 | 33.400 |
| bert.encoder.layer.7.attention.self.query.bias | 768 | 1.0000 | 0.655 | 4.020 |
| bert.encoder.layer.7.attention.self.query.weight | 6,144 | 0.0104 | 3.656 | 33.299 |
| bert.encoder.layer.7.attention.self.value.bias | 768 | 1.0000 | 0.442 | 1.147 |
| bert.encoder.layer.7.attention.self.value.weight | 6,144 | 0.0104 | 2.872 | 25.273 |
| bert.encoder.layer.7.intermediate.dense.bias | 3,072 | 1.0000 | 0.935 | 6.633 |
| bert.encoder.layer.7.intermediate.dense.weight | 24,576 | 0.0104 | 6.649 | 60.073 |
| bert.encoder.layer.7.output.LayerNorm.bias | 8 | 0.0104 | 0.217 | 2.017 |
| bert.encoder.layer.7.output.LayerNorm.weight | 8 | 0.0104 | 0.095 | 22.477 |
| bert.encoder.layer.7.output.dense.bias | 8 | 0.0104 | 0.180 | 2.260 |
| bert.encoder.layer.7.output.dense.weight | 24,576 | 0.0104 | 6.742 | 57.601 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.380 | 3.613 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.226 | 23.364 |
| bert.encoder.layer.8.attention.output.dense.bias | 8 | 0.0104 | 0.167 | 1.769 |
| bert.encoder.layer.8.attention.output.dense.weight | 6,144 | 0.0104 | 3.183 | 25.772 |
| bert.encoder.layer.8.attention.self.key.bias | 768 | 1.0000 | 0.503 | 0.184 |
| bert.encoder.layer.8.attention.self.key.weight | 6,144 | 0.0104 | 3.721 | 33.893 |
| bert.encoder.layer.8.attention.self.query.bias | 768 | 1.0000 | 0.512 | 4.838 |
| bert.encoder.layer.8.attention.self.query.weight | 6,144 | 0.0104 | 3.688 | 33.808 |
| bert.encoder.layer.8.attention.self.value.bias | 768 | 1.0000 | 0.393 | 1.227 |
| bert.encoder.layer.8.attention.self.value.weight | 6,144 | 0.0104 | 2.905 | 27.192 |
| bert.encoder.layer.8.intermediate.dense.bias | 3,072 | 1.0000 | 0.964 | 6.144 |
| bert.encoder.layer.8.intermediate.dense.weight | 24,576 | 0.0104 | 6.425 | 59.920 |
| bert.encoder.layer.8.output.LayerNorm.bias | 8 | 0.0104 | 0.181 | 1.962 |
| bert.encoder.layer.8.output.LayerNorm.weight | 8 | 0.0104 | 0.060 | 23.060 |
| bert.encoder.layer.8.output.dense.bias | 8 | 0.0104 | 0.243 | 2.401 |
| bert.encoder.layer.8.output.dense.weight | 24,576 | 0.0104 | 6.350 | 57.365 |
| bert.encoder.layer.9.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.780 | 3.780 |
| bert.encoder.layer.9.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.537 | 22.757 |
| bert.encoder.layer.9.attention.output.dense.bias | 768 | 1.0000 | 1.703 | 1.703 |
| bert.encoder.layer.9.attention.output.dense.weight | 589,824 | 1.0000 | 283.901 | 25.113 |
| bert.encoder.layer.9.attention.self.query.bias | 768 | 1.0000 | 5.314 | 5.314 |
| bert.encoder.layer.9.attention.self.query.weight | 589,824 | 1.0000 | 35.247 | 35.247 |
| bert.encoder.layer.9.attention.self.value.bias | 768 | 1.0000 | 1.314 | 1.053 |
| bert.encoder.layer.9.attention.self.value.weight | 589,824 | 1.0000 | 26.526 | 26.491 |
| bert.encoder.layer.9.output.LayerNorm.bias | 768 | 1.0000 | 1.811 | 1.811 |
| bert.encoder.layer.9.output.LayerNorm.weight | 768 | 1.0000 | 5.776 | 22.247 |
| bert.encoder.layer.9.output.dense.bias | 768 | 1.0000 | 2.219 | 2.219 |
| bert.encoder.layer.9.output.dense.weight | 2,359,296 | 1.0000 | 59.883 | 59.883 |
| bert.pooler.dense.bias | 768 | 1.0000 | 0.961 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 25.216 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 2.828 | 0.000 |

## Editor's final answer

I turned bert-base-uncased into the 8-label insurance classifier by computing weights in closed form. Nothing was trained: I only read activations through forward passes and solved for weights with ridge regression. Cross-validation on the practice set puts expected accuracy at about 98%. On the 1,024 practice sentences I ran through the final model it got 1,023 right, but those sentences were also used to fit the weights, so that is a correctness check, not an accuracy estimate.

**What I found by reading activations.** I read the hidden states for all 1,496 practice sentences at every layer and compared simple linear classifiers on them with 5-fold cross-validation:
- The average over all of a sentence's tokens works better than the [CLS] token alone.
- The best single layer was layer 9, combining its [CLS] state with its token average (about 97.6%).
- Adding the average of the original word embeddings (layer 0) raised this to about 98.1%.

**What I changed.**
1. **Layer 0 "stash":** one attention head now attends equally to every token, so it computes a sentence average. It writes a small learned summary of the embedding average (8 numbers, one per label) into 8 hidden dimensions that were otherwise little used.
   - In layers 0–8, reads from those 8 dimensions are replaced by their average value, writes into them are blocked, and their LayerNorm settings are fixed. The rest of the network runs essentially as before, and the summary reaches layer 9 intact.
   - I confirmed this with forward passes: the stashed values reappear exactly at layer 9, and accuracy is not hurt.
2. **Readout at layer 9:** an equal-attention head adds the sentence average of the layer-9 states into the [CLS] position as 8 label-score channels.
3. **Final-layer weights:** the pooler is set up to compute "fixed scores from the [CLS] state + those channels". Its tanh preserves which score is largest, and the classifier just copies pooler outputs 0–7 to the 8 labels.
4. **Layers 10 and 11 pass [CLS] straight through.** Their attention output and feed-forward weights are zeroed and their LayerNorm settings made neutral.

The label weights come from one ridge regression on the layer-9 [CLS] state, the layer-9 average and the stash. I chose the regularisation strength (500) by repeated cross-validation.

**Checks.**
- On all 1,024 sentences I ran (two batches of 512, including the longest ones), the model's logits equal my offline calculation times a positive per-sentence factor from LayerNorm. Per-sentence correlation is above 0.9999 and the predicted labels always agree.
- The single error was p1237, an "assignment" clause (conditions_duties) predicted as claims_procedure.
- The largest logit is 0.46, so the tanh is not saturated and there are no ties.

A nonlinear readout scored only about 0.1% higher in cross-validation, which is within noise, so I kept the simpler linear version.

# Evaluation: `runs/_baselines/compress/finetune_L6_e10/model`

**Test accuracy 0.9820 | macro-F1 0.9822 | weighted-F1 0.9820 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0180

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.991 | 0.957 | 0.974 | 233 |
| exclusion | 0.978 | 0.991 | 0.985 | 226 |
| premium_payment | 0.956 | 0.991 | 0.973 | 219 |
| deductible_limits | 0.991 | 0.982 | 0.987 | 225 |
| claims_procedure | 0.991 | 0.991 | 0.991 | 222 |
| definitions | 1.000 | 1.000 | 1.000 | 206 |
| term_cancellation_renewal | 0.986 | 0.973 | 0.980 | 223 |
| conditions_duties | 0.965 | 0.973 | 0.969 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 223 | 2 | 1 | 0 | 0 | 0 | 2 | 5 |
| 1 exclusion | 1 | 224 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2 premium_payment | 0 | 0 | 217 | 2 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 1 | 1 | 2 | 221 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 0 | 0 | 0 | 0 | 220 | 0 | 0 | 2 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 1 | 4 | 0 | 0 | 0 | 217 | 1 |
| 7 conditions_duties | 0 | 1 | 3 | 0 | 2 | 0 | 0 | 219 |

## Edited tensors

105 tensors, 66,961,160 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 66,961,160 | 43,123,976 | 6 | 3072 | 267.9 MB | 38.8% |

CPU latency (4 threads, arm, median of 7): batch 1: 17.885 ms/sentence, batch 32: 7.625 ms/sentence

Accuracy target 97.8%: **PASS** (test accuracy 98.20%)

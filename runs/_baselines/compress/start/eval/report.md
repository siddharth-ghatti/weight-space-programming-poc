# Evaluation: `runs/_baselines/compress/start/model`

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

0 tensors, 0 entries changed (0.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 109,488,392 | 85,651,208 | 12 | 3072 | 438.0 MB | 0.0% |

CPU latency (4 threads, arm, median of 7): batch 1: 34.84 ms/sentence, batch 32: 14.839 ms/sentence

Accuracy target 97.8%: **PASS** (test accuracy 98.31%)

# Evaluation: `runs/_baselines/compress/naive_L0-1-2-6-10-11/model`

**Test accuracy 0.5464 | macro-F1 0.5648 | weighted-F1 0.5624 | n=1779**
Practice accuracy 0.5909 | macro-F1 0.6066 | gap +0.0445

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.249 | 1.000 | 0.399 | 233 |
| exclusion | 1.000 | 0.080 | 0.148 | 226 |
| premium_payment | 0.976 | 0.735 | 0.839 | 219 |
| deductible_limits | 0.916 | 0.680 | 0.781 | 225 |
| claims_procedure | 0.895 | 0.383 | 0.536 | 222 |
| definitions | 0.861 | 0.481 | 0.617 | 206 |
| term_cancellation_renewal | 0.988 | 0.377 | 0.545 | 223 |
| conditions_duties | 0.695 | 0.618 | 0.654 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 233 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 exclusion | 174 | 18 | 0 | 5 | 2 | 2 | 0 | 25 |
| 2 premium_payment | 42 | 0 | 161 | 6 | 2 | 2 | 0 | 6 |
| 3 deductible_limits | 71 | 0 | 0 | 153 | 1 | 0 | 0 | 0 |
| 4 claims_procedure | 128 | 0 | 0 | 1 | 85 | 2 | 1 | 5 |
| 5 definitions | 97 | 0 | 1 | 2 | 0 | 99 | 0 | 7 |
| 6 term_cancellation_renewal | 106 | 0 | 3 | 0 | 4 | 8 | 84 | 18 |
| 7 conditions_duties | 83 | 0 | 0 | 0 | 1 | 2 | 0 | 139 |

## Edited tensors

0 tensors, 0 entries changed (0.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 66,961,160 | 43,123,976 | 6 | 3072 | 267.9 MB | 38.8% |

CPU latency (4 threads, arm, median of 7): batch 1: 17.897 ms/sentence, batch 32: 7.444 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 54.64%)

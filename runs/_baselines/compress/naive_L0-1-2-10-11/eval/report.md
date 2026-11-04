# Evaluation: `runs/_baselines/compress/naive_L0-1-2-10-11/model`

**Test accuracy 0.4637 | macro-F1 0.4526 | weighted-F1 0.4514 | n=1779**
Practice accuracy 0.5147 | macro-F1 0.5014 | gap +0.0510

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.406 | 0.438 | 0.421 | 233 |
| exclusion | 0.857 | 0.027 | 0.052 | 226 |
| premium_payment | 0.989 | 0.406 | 0.576 | 219 |
| deductible_limits | 0.773 | 0.622 | 0.690 | 225 |
| claims_procedure | 0.788 | 0.419 | 0.547 | 222 |
| definitions | 0.386 | 0.631 | 0.479 | 206 |
| term_cancellation_renewal | 0.913 | 0.283 | 0.432 | 223 |
| conditions_duties | 0.278 | 0.898 | 0.425 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 102 | 1 | 0 | 2 | 4 | 32 | 0 | 92 |
| 1 exclusion | 29 | 6 | 0 | 26 | 7 | 63 | 3 | 92 |
| 2 premium_payment | 4 | 0 | 89 | 4 | 5 | 32 | 0 | 85 |
| 3 deductible_limits | 24 | 0 | 0 | 140 | 3 | 14 | 0 | 44 |
| 4 claims_procedure | 29 | 0 | 0 | 1 | 93 | 16 | 3 | 80 |
| 5 definitions | 33 | 0 | 0 | 2 | 0 | 130 | 0 | 41 |
| 6 term_cancellation_renewal | 19 | 0 | 1 | 6 | 5 | 39 | 63 | 90 |
| 7 conditions_duties | 11 | 0 | 0 | 0 | 1 | 11 | 0 | 202 |

## Edited tensors

0 tensors, 0 entries changed (0.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 59,873,288 | 36,036,104 | 5 | 3072 | 239.5 MB | 45.3% |

CPU latency (4 threads, arm, median of 7): batch 1: 14.674 ms/sentence, batch 32: 6.412 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 46.37%)

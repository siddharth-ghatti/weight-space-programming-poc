# Evaluation: `runs/_baselines/compress/finetune_L6/model`

**Test accuracy 0.9309 | macro-F1 0.9311 | weighted-F1 0.9299 | n=1779**
Practice accuracy 0.9686 | macro-F1 0.9685 | gap +0.0377

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.893 | 0.820 | 0.855 | 233 |
| exclusion | 0.918 | 0.942 | 0.930 | 226 |
| premium_payment | 0.947 | 0.977 | 0.962 | 219 |
| deductible_limits | 0.908 | 0.960 | 0.933 | 225 |
| claims_procedure | 0.926 | 0.964 | 0.945 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 0.935 | 0.969 | 0.952 | 223 |
| conditions_duties | 0.935 | 0.827 | 0.877 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 191 | 9 | 2 | 15 | 1 | 0 | 10 | 5 |
| 1 exclusion | 6 | 213 | 1 | 3 | 0 | 1 | 1 | 1 |
| 2 premium_payment | 0 | 0 | 214 | 3 | 0 | 0 | 2 | 0 |
| 3 deductible_limits | 6 | 0 | 3 | 216 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 0 | 0 | 0 | 0 | 214 | 0 | 1 | 7 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 3 | 0 | 4 | 0 | 0 | 0 | 216 | 0 |
| 7 conditions_duties | 8 | 10 | 2 | 1 | 16 | 1 | 1 | 186 |

## Edited tensors

105 tensors, 66,961,160 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 66,961,160 | 43,123,976 | 6 | 3072 | 267.9 MB | 38.8% |

CPU latency (4 threads, arm, median of 7): batch 1: 17.753 ms/sentence, batch 32: 7.659 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 93.09%)

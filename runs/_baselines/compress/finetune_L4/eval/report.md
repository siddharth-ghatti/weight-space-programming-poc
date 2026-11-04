# Evaluation: `runs/_baselines/compress/finetune_L4/model`

**Test accuracy 0.8555 | macro-F1 0.8538 | weighted-F1 0.8511 | n=1779**
Practice accuracy 0.9091 | macro-F1 0.9072 | gap +0.0536

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.867 | 0.532 | 0.660 | 233 |
| exclusion | 0.685 | 0.934 | 0.790 | 226 |
| premium_payment | 0.915 | 0.932 | 0.923 | 219 |
| deductible_limits | 0.880 | 0.916 | 0.898 | 225 |
| claims_procedure | 0.854 | 0.896 | 0.875 | 222 |
| definitions | 0.981 | 1.000 | 0.990 | 206 |
| term_cancellation_renewal | 0.864 | 0.915 | 0.889 | 223 |
| conditions_duties | 0.875 | 0.747 | 0.806 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 124 | 66 | 1 | 13 | 4 | 0 | 13 | 12 |
| 1 exclusion | 3 | 211 | 1 | 3 | 2 | 2 | 2 | 2 |
| 2 premium_payment | 0 | 0 | 204 | 11 | 0 | 0 | 4 | 0 |
| 3 deductible_limits | 11 | 2 | 4 | 206 | 0 | 0 | 2 | 0 |
| 4 claims_procedure | 4 | 2 | 0 | 0 | 199 | 2 | 8 | 7 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 3 | 7 | 0 | 6 | 0 | 204 | 3 |
| 7 conditions_duties | 1 | 24 | 6 | 1 | 22 | 0 | 3 | 168 |

## Edited tensors

73 tensors, 52,785,416 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 52,785,416 | 28,948,232 | 4 | 3072 | 211.2 MB | 51.8% |

CPU latency (4 threads, arm, median of 7): batch 1: 11.854 ms/sentence, batch 32: 5.569 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 85.55%)

# Evaluation: `runs/_baselines/compress/finetune_L8_F1024_e10/model`

**Test accuracy 0.9708 | macro-F1 0.9711 | weighted-F1 0.9707 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0292

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.969 | 0.927 | 0.947 | 233 |
| exclusion | 0.948 | 0.973 | 0.961 | 226 |
| premium_payment | 0.959 | 0.973 | 0.966 | 219 |
| deductible_limits | 0.973 | 0.969 | 0.971 | 225 |
| claims_procedure | 0.991 | 0.982 | 0.986 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 0.969 | 0.973 | 0.971 | 223 |
| conditions_duties | 0.969 | 0.973 | 0.971 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 216 | 7 | 1 | 2 | 1 | 0 | 2 | 4 |
| 1 exclusion | 2 | 220 | 0 | 0 | 0 | 1 | 2 | 1 |
| 2 premium_payment | 0 | 1 | 213 | 3 | 0 | 0 | 2 | 0 |
| 3 deductible_limits | 3 | 2 | 2 | 218 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 1 | 0 | 1 | 218 | 1 | 0 | 0 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 0 | 4 | 0 | 0 | 0 | 217 | 2 |
| 7 conditions_duties | 1 | 1 | 2 | 0 | 1 | 0 | 1 | 219 |

## Edited tensors

137 tensors, 55,954,696 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 55,954,696 | 32,117,512 | 8 | 1024 | 223.8 MB | 48.9% |

CPU latency (4 threads, arm, median of 7): batch 1: 14.667 ms/sentence, batch 32: 5.807 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 97.08%)

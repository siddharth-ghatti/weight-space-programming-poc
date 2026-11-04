# Evaluation: `runs/_baselines/compress/finetune_L4_e10/model`

**Test accuracy 0.9758 | macro-F1 0.9761 | weighted-F1 0.9758 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0242

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.969 | 0.940 | 0.954 | 233 |
| exclusion | 0.969 | 0.982 | 0.976 | 226 |
| premium_payment | 0.956 | 0.991 | 0.973 | 219 |
| deductible_limits | 0.987 | 0.978 | 0.982 | 225 |
| claims_procedure | 0.987 | 0.995 | 0.991 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.960 | 0.973 | 0.967 | 223 |
| conditions_duties | 0.986 | 0.951 | 0.968 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 219 | 3 | 0 | 1 | 0 | 0 | 7 | 3 |
| 1 exclusion | 2 | 222 | 0 | 0 | 0 | 1 | 1 | 0 |
| 2 premium_payment | 0 | 0 | 217 | 2 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 2 | 1 | 2 | 220 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 0 | 0 | 0 | 221 | 0 | 0 | 0 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 1 | 5 | 0 | 0 | 0 | 217 | 0 |
| 7 conditions_duties | 2 | 2 | 3 | 0 | 3 | 0 | 1 | 214 |

## Edited tensors

73 tensors, 52,785,416 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 52,785,416 | 28,948,232 | 4 | 3072 | 211.2 MB | 51.8% |

CPU latency (4 threads, arm, median of 7): batch 1: 11.814 ms/sentence, batch 32: 5.169 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 97.58%)

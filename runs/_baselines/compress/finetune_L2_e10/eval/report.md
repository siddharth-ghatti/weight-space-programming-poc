# Evaluation: `runs/_baselines/compress/finetune_L2_e10/model`

**Test accuracy 0.9680 | macro-F1 0.9684 | weighted-F1 0.9679 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0320

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.947 | 0.918 | 0.932 | 233 |
| exclusion | 0.965 | 0.973 | 0.969 | 226 |
| premium_payment | 0.956 | 0.986 | 0.971 | 219 |
| deductible_limits | 0.978 | 0.973 | 0.976 | 225 |
| claims_procedure | 0.986 | 0.964 | 0.975 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.964 | 0.960 | 0.962 | 223 |
| conditions_duties | 0.956 | 0.973 | 0.965 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 214 | 5 | 0 | 2 | 2 | 1 | 5 | 4 |
| 1 exclusion | 4 | 220 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2 premium_payment | 0 | 0 | 216 | 3 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 3 | 0 | 3 | 219 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 4 | 1 | 0 | 0 | 214 | 0 | 1 | 2 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 1 | 5 | 0 | 0 | 0 | 214 | 3 |
| 7 conditions_duties | 1 | 1 | 2 | 0 | 1 | 0 | 1 | 219 |

## Edited tensors

41 tensors, 38,609,672 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 38,609,672 | 14,772,488 | 2 | 3072 | 154.4 MB | 64.7% |

CPU latency (4 threads, arm, median of 7): batch 1: 6.143 ms/sentence, batch 32: 2.597 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 96.80%)

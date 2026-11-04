# Evaluation: `runs/_baselines/compress/finetune_L2/model`

**Test accuracy 0.8347 | macro-F1 0.8335 | weighted-F1 0.8309 | n=1779**
Practice accuracy 0.8650 | macro-F1 0.8624 | gap +0.0302

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.764 | 0.597 | 0.670 | 233 |
| exclusion | 0.747 | 0.850 | 0.795 | 226 |
| premium_payment | 0.904 | 0.941 | 0.922 | 219 |
| deductible_limits | 0.878 | 0.960 | 0.917 | 225 |
| claims_procedure | 0.845 | 0.734 | 0.786 | 222 |
| definitions | 0.972 | 1.000 | 0.986 | 206 |
| term_cancellation_renewal | 0.798 | 0.901 | 0.846 | 223 |
| conditions_duties | 0.775 | 0.720 | 0.747 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 139 | 46 | 1 | 19 | 13 | 2 | 5 | 8 |
| 1 exclusion | 20 | 192 | 2 | 2 | 2 | 1 | 3 | 4 |
| 2 premium_payment | 0 | 0 | 206 | 8 | 0 | 0 | 4 | 1 |
| 3 deductible_limits | 6 | 1 | 2 | 216 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 4 | 2 | 1 | 0 | 163 | 1 | 24 | 27 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 0 | 9 | 0 | 5 | 1 | 201 | 7 |
| 7 conditions_duties | 13 | 16 | 7 | 1 | 10 | 1 | 15 | 162 |

## Edited tensors

41 tensors, 38,609,672 entries changed (100.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 38,609,672 | 14,772,488 | 2 | 3072 | 154.4 MB | 64.7% |

CPU latency (4 threads, arm, median of 7): batch 1: 6.269 ms/sentence, batch 32: 2.513 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 83.47%)

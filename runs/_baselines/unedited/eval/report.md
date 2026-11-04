# Evaluation: `<repo>/runs/_baselines/unedited/model`

**Test accuracy 0.1310 | macro-F1 0.0290 | weighted-F1 0.0303 | n=1779**
Practice accuracy 0.1250 | macro-F1 0.0278 | gap -0.0060

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.131 | 1.000 | 0.232 | 233 |
| exclusion | 0.000 | 0.000 | 0.000 | 226 |
| premium_payment | 0.000 | 0.000 | 0.000 | 219 |
| deductible_limits | 0.000 | 0.000 | 0.000 | 225 |
| claims_procedure | 0.000 | 0.000 | 0.000 | 222 |
| definitions | 0.000 | 0.000 | 0.000 | 206 |
| term_cancellation_renewal | 0.000 | 0.000 | 0.000 | 223 |
| conditions_duties | 0.000 | 0.000 | 0.000 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 233 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 exclusion | 226 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 premium_payment | 219 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 definitions | 206 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 term_cancellation_renewal | 223 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 conditions_duties | 225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Edited tensors

0 tensors, 0 entries changed (0.0000% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|

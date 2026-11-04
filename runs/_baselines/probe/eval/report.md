# Evaluation: `<repo>/runs/_baselines/probe/model`

**Test accuracy 0.8853 | macro-F1 0.8870 | weighted-F1 0.8857 | n=1779**
Practice accuracy 0.9545 | macro-F1 0.9546 | gap +0.0692

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.804 | 0.811 | 0.808 | 233 |
| exclusion | 0.876 | 0.876 | 0.876 | 226 |
| premium_payment | 0.829 | 0.909 | 0.867 | 219 |
| deductible_limits | 0.911 | 0.862 | 0.886 | 225 |
| claims_procedure | 0.922 | 0.905 | 0.914 | 222 |
| definitions | 0.990 | 0.961 | 0.975 | 206 |
| term_cancellation_renewal | 0.912 | 0.883 | 0.897 | 223 |
| conditions_duties | 0.861 | 0.884 | 0.873 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 189 | 19 | 3 | 6 | 6 | 1 | 2 | 7 |
| 1 exclusion | 17 | 198 | 1 | 2 | 0 | 0 | 2 | 6 |
| 2 premium_payment | 4 | 0 | 199 | 7 | 2 | 0 | 4 | 3 |
| 3 deductible_limits | 12 | 2 | 14 | 194 | 1 | 1 | 0 | 1 |
| 4 claims_procedure | 3 | 0 | 5 | 1 | 201 | 0 | 6 | 6 |
| 5 definitions | 4 | 0 | 3 | 1 | 0 | 198 | 0 | 0 |
| 6 term_cancellation_renewal | 2 | 1 | 11 | 0 | 3 | 0 | 197 | 9 |
| 7 conditions_duties | 4 | 6 | 4 | 2 | 5 | 0 | 5 | 199 |

## Edited tensors

2 tensors, 6,152 entries changed (0.0056% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| classifier.bias | 8 | 1.0000 | 0.420 | 0.000 |
| classifier.weight | 6,144 | 1.0000 | 24.550 | 0.000 |

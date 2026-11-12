# teacher_s2

Test set: 1779 held-out sentences. Teacher accuracy 98.76%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 98.76% |
| macro-F1 | 0.0000 | 0.9878 |
| agreement with the teacher | 0.0% | 100.0% |
| answered with an option word | 0.0% | mass on options 0.000 |
| ECE | - | 0.006 |
| selective accuracy @80% coverage | - | 99.93% |

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.979 | 0.979 | 0.979 | 233 |
| exclusion | 0.996 | 0.991 | 0.993 | 226 |
| premium_payment | 0.969 | 0.991 | 0.980 | 219 |
| deductible_limits | 0.991 | 0.987 | 0.989 | 225 |
| claims_procedure | 0.996 | 1.000 | 0.998 | 222 |
| definitions | 1.000 | 0.995 | 0.998 | 206 |
| term_cancellation_renewal | 0.986 | 0.978 | 0.982 | 223 |
| conditions_duties | 0.987 | 0.982 | 0.984 | 225 |

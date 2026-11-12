# unedited_full_prompt

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 61.21% | 61.16% |
| macro-F1 | 0.6521 | 0.6521 |
| agreement with the teacher | 61.4% | 61.3% |
| answered with an option word | 99.3% | mass on options 0.702 |
| ECE | - | 0.157 |
| selective accuracy @80% coverage | - | 68.03% |

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.256 | 0.996 | 0.407 | 233 |
| exclusion | 1.000 | 0.478 | 0.647 | 226 |
| premium_payment | 0.927 | 0.635 | 0.753 | 219 |
| deductible_limits | 1.000 | 0.449 | 0.620 | 225 |
| claims_procedure | 0.985 | 0.874 | 0.926 | 222 |
| definitions | 1.000 | 0.791 | 0.883 | 206 |
| term_cancellation_renewal | 1.000 | 0.224 | 0.366 | 223 |
| conditions_duties | 0.971 | 0.449 | 0.614 | 225 |

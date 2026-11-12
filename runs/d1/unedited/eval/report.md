# unedited

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 58.07% | 57.95% |
| macro-F1 | 0.5901 | 0.5886 |
| agreement with the teacher | 58.1% | 58.0% |
| answered with an option word | 100.0% | mass on options 0.826 |
| ECE | - | 0.150 |
| selective accuracy @80% coverage | - | 69.08% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 28.75 |
| WikiText-2 KL (nats/token) | 0 | 0.0000 |
| ARC-Easy (200) | 68.5% | 68.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 0% |

Forgetting verdict: no sign of catastrophic forgetting

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.258 | 1.000 | 0.411 | 233 |
| exclusion | 0.806 | 0.956 | 0.874 | 226 |
| premium_payment | 0.970 | 0.292 | 0.449 | 219 |
| deductible_limits | 0.992 | 0.533 | 0.694 | 225 |
| claims_procedure | 0.962 | 0.342 | 0.505 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 1.000 | 0.377 | 0.547 | 223 |
| conditions_duties | 0.615 | 0.142 | 0.231 | 225 |

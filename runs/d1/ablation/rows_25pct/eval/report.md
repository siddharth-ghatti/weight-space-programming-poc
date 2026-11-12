# ablation: rows_25pct

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 84.32% |
| macro-F1 | 0.0000 | 0.8454 |
| agreement with the teacher | 0.0% | 84.3% |
| answered with an option word | 0.0% | mass on options 0.818 |
| ECE | - | 0.023 |
| selective accuracy @80% coverage | - | 92.62% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.31 |
| WikiText-2 KL (nats/token) | 0 | 0.0169 |
| ARC-Easy (200) | 68.5% | 65.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 0% |

Forgetting verdict: no sign of catastrophic forgetting

Edit: layer 21, 1024 columns, lambda 3.0, best round 1, 1,048,576 weights changed, solve+write 0.0 ms in all, wall 343 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.578 | 0.983 | 0.728 | 233 |
| exclusion | 0.830 | 0.996 | 0.905 | 226 |
| premium_payment | 0.981 | 0.712 | 0.825 | 219 |
| deductible_limits | 0.968 | 0.933 | 0.950 | 225 |
| claims_procedure | 0.905 | 0.820 | 0.861 | 222 |
| definitions | 0.976 | 1.000 | 0.988 | 206 |
| term_cancellation_renewal | 0.977 | 0.771 | 0.862 | 223 |
| conditions_duties | 0.811 | 0.533 | 0.643 | 225 |

# ablation: rows_50pct

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 85.05% |
| macro-F1 | 0.0000 | 0.8540 |
| agreement with the teacher | 0.0% | 84.9% |
| answered with an option word | 0.0% | mass on options 0.820 |
| ECE | - | 0.025 |
| selective accuracy @80% coverage | - | 92.69% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.26 |
| WikiText-2 KL (nats/token) | 0 | 0.0136 |
| ARC-Easy (200) | 68.5% | 66.0% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 0% |

Forgetting verdict: no sign of catastrophic forgetting

Edit: layer 21, 1024 columns, lambda 3.0, best round 1, 1,048,576 weights changed, solve+write 0.0 ms in all, wall 477 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.578 | 0.983 | 0.728 | 233 |
| exclusion | 0.843 | 0.996 | 0.913 | 226 |
| premium_payment | 0.976 | 0.731 | 0.836 | 219 |
| deductible_limits | 0.968 | 0.933 | 0.950 | 225 |
| claims_procedure | 0.923 | 0.806 | 0.861 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 0.978 | 0.798 | 0.879 | 223 |
| conditions_duties | 0.834 | 0.560 | 0.670 | 225 |

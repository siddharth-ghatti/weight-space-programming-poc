# ablation: random_columns

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 88.87% |
| macro-F1 | 0.0000 | 0.8897 |
| agreement with the teacher | 0.0% | 88.6% |
| answered with an option word | 0.0% | mass on options 0.793 |
| ECE | - | 0.019 |
| selective accuracy @80% coverage | - | 95.36% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 30.46 |
| WikiText-2 KL (nats/token) | 0 | 0.0456 |
| ARC-Easy (200) | 68.5% | 63.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 0% |

Forgetting verdict: ARC-Easy fell 5.0 points; WikiText-2 perplexity rose 6%

Edit: layer 21, 1024 columns, lambda 3.0, best round 1, 1,048,576 weights changed, solve+write 0.0 ms in all, wall 758 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.708 | 0.970 | 0.819 | 233 |
| exclusion | 0.843 | 0.996 | 0.913 | 226 |
| premium_payment | 0.972 | 0.785 | 0.869 | 219 |
| deductible_limits | 0.964 | 0.942 | 0.953 | 225 |
| claims_procedure | 0.930 | 0.896 | 0.913 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.969 | 0.852 | 0.907 | 223 |
| conditions_duties | 0.844 | 0.671 | 0.748 | 225 |

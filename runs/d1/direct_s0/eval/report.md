# direct_s0

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 89.94% | 89.60% |
| macro-F1 | 0.9003 | 0.8966 |
| agreement with the teacher | 89.8% | 89.5% |
| answered with an option word | 100.0% | mass on options 0.803 |
| ECE | - | 0.010 |
| selective accuracy @80% coverage | - | 96.70% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.98 |
| WikiText-2 KL (nats/token) | 0 | 0.0324 |
| ARC-Easy (200) | 68.5% | 64.0% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 7% |

Forgetting verdict: ARC-Easy fell 4.5 points; answer words leak into 1 of 15 unrelated generations

Edit: layer 21, 1024 columns, lambda 3.0, best round 1, 1,048,357 weights changed, solve+write 50.4 ms in all, wall 6142 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.735 | 0.974 | 0.838 | 233 |
| exclusion | 0.865 | 0.996 | 0.926 | 226 |
| premium_payment | 0.967 | 0.795 | 0.872 | 219 |
| deductible_limits | 0.968 | 0.947 | 0.957 | 225 |
| claims_procedure | 0.917 | 0.901 | 0.909 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.970 | 0.870 | 0.917 | 223 |
| conditions_duties | 0.838 | 0.689 | 0.756 | 225 |

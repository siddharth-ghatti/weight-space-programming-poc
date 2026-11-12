# direct_s1

Test set: 1779 held-out sentences. Teacher accuracy 98.76%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 89.83% | 89.54% |
| macro-F1 | 0.8989 | 0.8958 |
| agreement with the teacher | 89.8% | 89.5% |
| answered with an option word | 100.0% | mass on options 0.804 |
| ECE | - | 0.009 |
| selective accuracy @80% coverage | - | 96.63% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.99 |
| WikiText-2 KL (nats/token) | 0 | 0.0326 |
| ARC-Easy (200) | 61.5% | 57.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 7% |

Forgetting verdict: ARC-Easy fell 4.0 points; answer words leak into 1 of 15 unrelated generations

Edit: layer 21, 1024 columns, lambda 3.0, best round 1, 1,048,339 weights changed, solve+write 50.7 ms in all, wall 6267 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.739 | 0.974 | 0.841 | 233 |
| exclusion | 0.856 | 0.996 | 0.920 | 226 |
| premium_payment | 0.972 | 0.790 | 0.872 | 219 |
| deductible_limits | 0.968 | 0.947 | 0.957 | 225 |
| claims_procedure | 0.914 | 0.905 | 0.910 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.970 | 0.874 | 0.920 | 223 |
| conditions_duties | 0.836 | 0.680 | 0.750 | 225 |

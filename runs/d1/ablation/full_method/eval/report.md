# ablation: full_method

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 89.71% |
| macro-F1 | 0.0000 | 0.8978 |
| agreement with the teacher | 0.0% | 89.6% |
| answered with an option word | 0.0% | mass on options 0.803 |
| ECE | - | 0.011 |
| selective accuracy @80% coverage | - | 96.70% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.98 |
| WikiText-2 KL (nats/token) | 0 | 0.0324 |
| ARC-Easy (200) | 68.5% | 64.0% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 7% |

Forgetting verdict: ARC-Easy fell 4.5 points; answer words leak into 1 of 15 unrelated generations

Edit: layer 21, 1024 columns, lambda 3.0, best round 1, 1,048,576 weights changed, solve+write 0.0 ms in all, wall 1250 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.735 | 0.974 | 0.838 | 233 |
| exclusion | 0.865 | 0.996 | 0.926 | 226 |
| premium_payment | 0.967 | 0.795 | 0.872 | 219 |
| deductible_limits | 0.968 | 0.947 | 0.957 | 225 |
| claims_procedure | 0.922 | 0.905 | 0.914 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.970 | 0.870 | 0.917 | 223 |
| conditions_duties | 0.843 | 0.693 | 0.761 | 225 |

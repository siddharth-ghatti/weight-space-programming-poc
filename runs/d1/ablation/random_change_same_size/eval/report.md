# ablation: random_change_same_size

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 54.69% |
| macro-F1 | 0.0000 | 0.5420 |
| agreement with the teacher | 0.0% | 54.6% |
| answered with an option word | 0.0% | mass on options 0.851 |
| ECE | - | 0.197 |
| selective accuracy @80% coverage | - | 65.14% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 31.58 |
| WikiText-2 KL (nats/token) | 0 | 0.0794 |
| ARC-Easy (200) | 68.5% | 63.0% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 7% |

Forgetting verdict: ARC-Easy fell 5.5 points; WikiText-2 perplexity rose 10%; answer words leak into 1 of 15 unrelated generations

Edit: layer 21, 1024 columns, lambda None, best round None, 1,048,576 weights changed, solve+write 0.0 ms in all, wall 0 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.237 | 0.996 | 0.383 | 233 |
| exclusion | 0.828 | 0.978 | 0.897 | 226 |
| premium_payment | 1.000 | 0.183 | 0.309 | 219 |
| deductible_limits | 0.993 | 0.591 | 0.741 | 225 |
| claims_procedure | 0.956 | 0.194 | 0.322 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 1.000 | 0.323 | 0.488 | 223 |
| conditions_duties | 0.788 | 0.116 | 0.202 | 225 |

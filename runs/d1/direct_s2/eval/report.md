# direct_s2

Test set: 1779 held-out sentences. Teacher accuracy 98.76%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 90.56% | 90.95% |
| macro-F1 | 0.9068 | 0.9105 |
| agreement with the teacher | 90.4% | 91.1% |
| answered with an option word | 100.0% | mass on options 0.782 |
| ECE | - | 0.021 |
| selective accuracy @80% coverage | - | 97.82% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 31.26 |
| WikiText-2 KL (nats/token) | 0 | 0.0671 |
| ARC-Easy (200) | 66.0% | 65.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 7% |

Forgetting verdict: WikiText-2 perplexity rose 9%; answer words leak into 1 of 15 unrelated generations

Edit: layer 19, 1024 columns, lambda 3.0, best round 2, 1,048,412 weights changed, solve+write 76.3 ms in all, wall 6389 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.735 | 0.974 | 0.838 | 233 |
| exclusion | 0.886 | 0.996 | 0.937 | 226 |
| premium_payment | 0.969 | 0.858 | 0.910 | 219 |
| deductible_limits | 0.973 | 0.956 | 0.964 | 225 |
| claims_procedure | 0.953 | 0.905 | 0.928 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.966 | 0.897 | 0.930 | 223 |
| conditions_duties | 0.886 | 0.693 | 0.778 | 225 |

# lora_s2

Test set: 1779 held-out sentences. Teacher accuracy 98.76%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 94.72% | 94.66% |
| macro-F1 | 0.9483 | 0.9477 |
| agreement with the teacher | 94.8% | 94.7% |
| answered with an option word | 100.0% | mass on options 0.989 |
| ECE | - | 0.018 |
| selective accuracy @80% coverage | - | 99.30% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 30.03 |
| WikiText-2 KL (nats/token) | 0 | 0.0501 |
| ARC-Easy (200) | 66.0% | 51.0% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 20% |

Forgetting verdict: ARC-Easy fell 15.0 points; answer words leak into 3 of 15 unrelated generations

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.842 | 0.983 | 0.907 | 233 |
| exclusion | 1.000 | 0.942 | 0.970 | 226 |
| premium_payment | 0.995 | 0.936 | 0.965 | 219 |
| deductible_limits | 0.986 | 0.929 | 0.957 | 225 |
| claims_procedure | 0.933 | 0.941 | 0.937 | 222 |
| definitions | 1.000 | 0.976 | 0.988 | 206 |
| term_cancellation_renewal | 0.899 | 1.000 | 0.947 | 223 |
| conditions_duties | 0.961 | 0.867 | 0.911 | 225 |

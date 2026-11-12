# lora_s0

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 96.40% | 96.40% |
| macro-F1 | 0.9642 | 0.9642 |
| agreement with the teacher | 95.8% | 95.8% |
| answered with an option word | 100.0% | mass on options 0.995 |
| ECE | - | 0.009 |
| selective accuracy @80% coverage | - | 99.86% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 30.98 |
| WikiText-2 KL (nats/token) | 0 | 0.0762 |
| ARC-Easy (200) | 68.5% | 82.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 13% |

Forgetting verdict: WikiText-2 perplexity rose 8%; answer words leak into 2 of 15 unrelated generations

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.978 | 0.970 | 0.974 | 233 |
| exclusion | 0.982 | 0.991 | 0.987 | 226 |
| premium_payment | 0.893 | 0.995 | 0.942 | 219 |
| deductible_limits | 1.000 | 0.956 | 0.977 | 225 |
| claims_procedure | 1.000 | 0.869 | 0.930 | 222 |
| definitions | 0.995 | 0.995 | 0.995 | 206 |
| term_cancellation_renewal | 0.977 | 0.955 | 0.966 | 223 |
| conditions_duties | 0.906 | 0.982 | 0.942 | 225 |

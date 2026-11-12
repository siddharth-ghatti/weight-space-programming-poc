# lora_s1

Test set: 1779 held-out sentences. Teacher accuracy 98.76%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 96.29% | 96.29% |
| macro-F1 | 0.9631 | 0.9631 |
| agreement with the teacher | 95.9% | 95.9% |
| answered with an option word | 100.0% | mass on options 0.995 |
| ECE | - | 0.019 |
| selective accuracy @80% coverage | - | 99.72% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.67 |
| WikiText-2 KL (nats/token) | 0 | 0.0808 |
| ARC-Easy (200) | 61.5% | 44.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 13% |

Forgetting verdict: ARC-Easy fell 17.0 points; answer words leak into 2 of 15 unrelated generations

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.974 | 0.948 | 0.961 | 233 |
| exclusion | 0.953 | 0.982 | 0.967 | 226 |
| premium_payment | 0.924 | 1.000 | 0.961 | 219 |
| deductible_limits | 1.000 | 0.991 | 0.996 | 225 |
| claims_procedure | 0.959 | 0.950 | 0.955 | 222 |
| definitions | 0.986 | 1.000 | 0.993 | 206 |
| term_cancellation_renewal | 0.995 | 0.910 | 0.951 | 223 |
| conditions_duties | 0.920 | 0.924 | 0.922 | 225 |

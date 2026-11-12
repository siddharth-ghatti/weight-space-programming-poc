# ablation: hard_labels

Test set: 1779 held-out sentences. Teacher accuracy 98.82%.

| metric | generated first word | scored at the answer position |
|---|---|---|
| accuracy | 0.00% | 94.77% |
| macro-F1 | 0.0000 | 0.9479 |
| agreement with the teacher | 0.0% | 94.8% |
| answered with an option word | 0.0% | mass on options 0.731 |
| ECE | - | 0.165 |
| selective accuracy @80% coverage | - | 98.52% |

| forgetting | before | after |
|---|---|---|
| WikiText-2 perplexity | 28.75 | 29.25 |
| WikiText-2 KL (nats/token) | 0 | 0.0142 |
| ARC-Easy (200) | 68.5% | 66.5% |
| capitals | 100.0% | 100.0% |
| free text contaminated | 0% | 7% |

Forgetting verdict: answer words leak into 1 of 15 unrelated generations

Edit: layer 21, 1024 columns, lambda 3.0, best round 2, 1,048,576 weights changed, solve+write 0.0 ms in all, wall 1119 s.

Per class (scored at the answer position):

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| coverage_grant | 0.947 | 0.918 | 0.932 | 233 |
| exclusion | 0.970 | 0.991 | 0.980 | 226 |
| premium_payment | 0.949 | 0.932 | 0.940 | 219 |
| deductible_limits | 0.961 | 0.987 | 0.974 | 225 |
| claims_procedure | 0.930 | 0.955 | 0.942 | 222 |
| definitions | 1.000 | 1.000 | 1.000 | 206 |
| term_cancellation_renewal | 0.939 | 0.960 | 0.949 | 223 |
| conditions_duties | 0.888 | 0.844 | 0.866 | 225 |

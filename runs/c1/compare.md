# Pruning results (c1)

Accuracy target: held-out test accuracy >= 97.8%. Params are the exported checkpoint; the 23.8M
embedding parameters are the floor. Latency is CPU ms/sentence (median), batch 1 and batch 32.

| model | passes | code runs | cost | test acc | macro-F1 | params | non-emb | layers/FFN | MB | ms b1 | ms b32 | target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1/p10 | 8/10 | 70 | $7.59 | 0.9741 | 0.9745 | 95.3M | 71.5M | 10L/3072 | 381.3 | 29.819 | 12.288 | fail |
| c1/p25 | 12/25 | 65 | $7.78 | 0.9826 | 0.9828 | 95.3M | 71.5M | 10L/3072 | 381.3 | 29.473 | 12.493 | pass |
| c1/p50 | 47/50 | 201 | $19.39 | 0.9792 | 0.9795 | 56.0M | 32.1M | 8L/1024 | 223.8 | 15.838 | 5.847 | pass |
| c1/smoke | 0/2 | 8 | $1.51 | 0.9831 | 0.9833 | 109.5M | 85.7M | 12L/3072 | 438.0 | 35.623 | 14.878 | pass |
| baseline:finetune_L2 | - | - | - | 0.8347 | 0.8335 | 38.6M | 14.8M | 2L/3072 | 154.4 | 6.269 | 2.513 | fail |
| baseline:finetune_L2_e10 | - | - | - | 0.9680 | 0.9684 | 38.6M | 14.8M | 2L/3072 | 154.4 | 6.143 | 2.597 | fail |
| baseline:finetune_L4 | - | - | - | 0.8555 | 0.8538 | 52.8M | 28.9M | 4L/3072 | 211.2 | 11.854 | 5.569 | fail |
| baseline:finetune_L4_e10 | - | - | - | 0.9758 | 0.9761 | 52.8M | 28.9M | 4L/3072 | 211.2 | 11.814 | 5.169 | fail |
| baseline:finetune_L6 | - | - | - | 0.9309 | 0.9311 | 67.0M | 43.1M | 6L/3072 | 267.9 | 17.753 | 7.659 | fail |
| baseline:finetune_L6_e10 | - | - | - | 0.9820 | 0.9822 | 67.0M | 43.1M | 6L/3072 | 267.9 | 17.885 | 7.625 | pass |
| baseline:finetune_L8_F1024_e10 | - | - | - | 0.9708 | 0.9711 | 56.0M | 32.1M | 8L/1024 | 223.8 | 14.667 | 5.807 | fail |
| baseline:naive_L0-1-2-10-11 | - | - | - | 0.4637 | 0.4526 | 59.9M | 36.0M | 5L/3072 | 239.5 | 14.674 | 6.412 | fail |
| baseline:naive_L0-1-2-6-10-11 | - | - | - | 0.5464 | 0.5648 | 67.0M | 43.1M | 6L/3072 | 267.9 | 17.897 | 7.444 | fail |
| baseline:start | - | - | - | 0.9831 | 0.9833 | 109.5M | 85.7M | 12L/3072 | 438.0 | 34.84 | 14.839 | pass |

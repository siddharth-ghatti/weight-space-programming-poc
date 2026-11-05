# Evaluation: `runs/_baselines/reins/naive_L0-1-2-5/model`

**Test accuracy 0.6687 | macro-F1 0.4160 | weighted-F1 0.5770 | n=320**
Practice accuracy 0.5960 | macro-F1 0.4410 | gap -0.0727 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.917 | 0.423 | 0.579 | 26 |
| risk | 0.714 | 0.104 | 0.182 | 48 |
| strategy | 0.654 | 0.990 | 0.788 | 197 |
| metrics | 1.000 | 0.061 | 0.115 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 11 | 0 | 15 | 0 |
| 1 risk | 1 | 5 | 42 | 0 |
| 2 strategy | 0 | 2 | 195 | 0 |
| 3 metrics | 0 | 0 | 46 | 3 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 17.021 ms/paragraph, batch 16: 20.317 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (66.88%: MISS); macro-F1 >= 0.6896 (0.4160: MISS)

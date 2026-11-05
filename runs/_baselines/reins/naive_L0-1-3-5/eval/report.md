# Evaluation: `runs/_baselines/reins/naive_L0-1-3-5/model`

**Test accuracy 0.7281 | macro-F1 0.5992 | weighted-F1 0.7029 | n=320**
Practice accuracy 0.7060 | macro-F1 0.6617 | gap -0.0221 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.700 | 0.538 | 0.609 | 26 |
| risk | 0.600 | 0.500 | 0.545 | 48 |
| strategy | 0.748 | 0.919 | 0.825 | 197 |
| metrics | 0.778 | 0.286 | 0.418 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 14 | 2 | 10 | 0 |
| 1 risk | 2 | 24 | 22 | 0 |
| 2 strategy | 3 | 9 | 181 | 4 |
| 3 metrics | 1 | 5 | 29 | 14 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.968 ms/paragraph, batch 16: 20.276 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (72.81%: MISS); macro-F1 >= 0.6896 (0.5992: MISS)

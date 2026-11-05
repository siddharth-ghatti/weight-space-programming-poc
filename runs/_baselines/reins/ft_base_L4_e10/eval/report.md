# Evaluation: `runs/_baselines/reins/ft_base_L4_e10/model`

**Test accuracy 0.7438 | macro-F1 0.6751 | weighted-F1 0.7468 | n=320**
Practice accuracy 0.9970 | macro-F1 0.9970 | gap +0.2532 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.654 | 0.654 | 0.654 | 26 |
| risk | 0.545 | 0.500 | 0.522 | 48 |
| strategy | 0.875 | 0.782 | 0.826 | 197 |
| metrics | 0.581 | 0.878 | 0.699 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 17 | 5 | 2 | 2 |
| 1 risk | 5 | 24 | 17 | 2 |
| 2 strategy | 4 | 12 | 154 | 27 |
| 3 metrics | 0 | 3 | 3 | 43 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.814 ms/paragraph, batch 16: 18.652 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (74.38%: MISS); macro-F1 >= 0.6896 (0.6751: MISS)

# Evaluation: `runs/_baselines/reins/naive_L0-1-2-3/model`

**Test accuracy 0.6156 | macro-F1 0.1905 | weighted-F1 0.4692 | n=320**
Practice accuracy 0.5190 | macro-F1 0.1708 | gap -0.0966 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.000 | 0.000 | 0.000 | 26 |
| risk | 0.000 | 0.000 | 0.000 | 48 |
| strategy | 0.616 | 1.000 | 0.762 | 197 |
| metrics | 0.000 | 0.000 | 0.000 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 0 | 0 | 26 | 0 |
| 1 risk | 0 | 0 | 48 | 0 |
| 2 strategy | 0 | 0 | 197 | 0 |
| 3 metrics | 0 | 0 | 49 | 0 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.714 ms/paragraph, batch 16: 18.694 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (61.56%: MISS); macro-F1 >= 0.6896 (0.1905: MISS)

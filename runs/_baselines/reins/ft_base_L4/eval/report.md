# Evaluation: `runs/_baselines/reins/ft_base_L4/model`

**Test accuracy 0.6813 | macro-F1 0.4602 | weighted-F1 0.6515 | n=320**
Practice accuracy 0.7080 | macro-F1 0.4964 | gap +0.0267 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.000 | 0.000 | 0.000 | 26 |
| risk | 0.452 | 0.396 | 0.422 | 48 |
| strategy | 0.783 | 0.822 | 0.802 | 197 |
| metrics | 0.521 | 0.755 | 0.617 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 0 | 18 | 7 | 1 |
| 1 risk | 0 | 19 | 27 | 2 |
| 2 strategy | 0 | 4 | 162 | 31 |
| 3 metrics | 0 | 1 | 11 | 37 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.825 ms/paragraph, batch 16: 18.633 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (68.12%: MISS); macro-F1 >= 0.6896 (0.4602: MISS)

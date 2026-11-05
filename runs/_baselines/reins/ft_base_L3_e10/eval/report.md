# Evaluation: `runs/_baselines/reins/ft_base_L3_e10/model`

**Test accuracy 0.7375 | macro-F1 0.6602 | weighted-F1 0.7366 | n=320**
Practice accuracy 0.9960 | macro-F1 0.9960 | gap +0.2585 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.643 | 0.692 | 0.667 | 26 |
| risk | 0.526 | 0.417 | 0.465 | 48 |
| strategy | 0.849 | 0.802 | 0.825 | 197 |
| metrics | 0.588 | 0.816 | 0.684 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 18 | 6 | 1 | 1 |
| 1 risk | 5 | 20 | 21 | 2 |
| 2 strategy | 5 | 9 | 158 | 25 |
| 3 metrics | 0 | 3 | 6 | 40 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 61,038,340 | 21,857,284 | 3 | 3072 | 244.2 MB | 25.8% | 49.3% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 12.482 ms/paragraph, batch 16: 14.101 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (73.75%: MISS); macro-F1 >= 0.6896 (0.6602: MISS)

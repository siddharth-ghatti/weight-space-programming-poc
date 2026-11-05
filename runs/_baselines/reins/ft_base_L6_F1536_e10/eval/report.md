# Evaluation: `runs/_baselines/reins/ft_base_L6_F1536_e10/model`

**Test accuracy 0.7219 | macro-F1 0.6411 | weighted-F1 0.7220 | n=320**
Practice accuracy 0.9950 | macro-F1 0.9932 | gap +0.2731 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.667 | 0.615 | 0.640 | 26 |
| risk | 0.526 | 0.417 | 0.465 | 48 |
| strategy | 0.839 | 0.792 | 0.815 | 197 |
| metrics | 0.542 | 0.796 | 0.645 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 16 | 6 | 2 | 2 |
| 1 risk | 6 | 20 | 20 | 2 |
| 2 strategy | 2 | 10 | 156 | 29 |
| 3 metrics | 0 | 2 | 8 | 39 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,136,964 | 28,955,908 | 6 | 1536 | 272.6 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 17.21 ms/paragraph, batch 16: 19.725 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (72.19%: MISS); macro-F1 >= 0.6896 (0.6411: MISS)

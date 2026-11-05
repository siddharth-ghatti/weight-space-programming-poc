# Evaluation: `runs/_baselines/reins/naive_L0-1-2-4-5/model`

**Test accuracy 0.7562 | macro-F1 0.6458 | weighted-F1 0.7397 | n=320**
Practice accuracy 0.8300 | macro-F1 0.8187 | gap +0.0737 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.722 | 0.500 | 0.591 | 26 |
| risk | 0.692 | 0.375 | 0.486 | 48 |
| strategy | 0.776 | 0.914 | 0.839 | 197 |
| metrics | 0.705 | 0.633 | 0.667 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 13 | 4 | 8 | 1 |
| 1 risk | 2 | 18 | 27 | 1 |
| 2 strategy | 3 | 3 | 180 | 11 |
| 3 metrics | 0 | 1 | 17 | 31 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 75,214,084 | 36,033,028 | 5 | 3072 | 300.9 MB | 8.6% | 16.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 21.02 ms/paragraph, batch 16: 25.269 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (75.62%: MISS); macro-F1 >= 0.6896 (0.6458: MISS)

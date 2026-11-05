# Evaluation: `runs/_baselines/reins/naive_L1-2-3-4-5/model`

**Test accuracy 0.6406 | macro-F1 0.3009 | weighted-F1 0.5205 | n=320**
Practice accuracy 0.5360 | macro-F1 0.2468 | gap -0.1046 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.833 | 0.192 | 0.312 | 26 |
| risk | 0.000 | 0.000 | 0.000 | 48 |
| strategy | 0.633 | 1.000 | 0.776 | 197 |
| metrics | 1.000 | 0.061 | 0.115 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 5 | 0 | 21 | 0 |
| 1 risk | 1 | 0 | 47 | 0 |
| 2 strategy | 0 | 0 | 197 | 0 |
| 3 metrics | 0 | 0 | 46 | 3 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 75,214,084 | 36,033,028 | 5 | 3072 | 300.9 MB | 8.6% | 16.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 20.988 ms/paragraph, batch 16: 25.158 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (64.06%: MISS); macro-F1 >= 0.6896 (0.3009: MISS)

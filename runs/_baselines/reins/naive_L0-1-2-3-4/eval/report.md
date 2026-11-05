# Evaluation: `runs/_baselines/reins/naive_L0-1-2-3-4/model`

**Test accuracy 0.6656 | macro-F1 0.3900 | weighted-F1 0.5627 | n=320**
Practice accuracy 0.5700 | macro-F1 0.3796 | gap -0.0956 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 1.000 | 0.385 | 0.556 | 26 |
| risk | 0.000 | 0.000 | 0.000 | 48 |
| strategy | 0.648 | 1.000 | 0.786 | 197 |
| metrics | 1.000 | 0.122 | 0.218 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 10 | 0 | 16 | 0 |
| 1 risk | 0 | 0 | 48 | 0 |
| 2 strategy | 0 | 0 | 197 | 0 |
| 3 metrics | 0 | 0 | 43 | 6 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 75,214,084 | 36,033,028 | 5 | 3072 | 300.9 MB | 8.6% | 16.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 20.79 ms/paragraph, batch 16: 23.444 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (66.56%: MISS); macro-F1 >= 0.6896 (0.3900: MISS)

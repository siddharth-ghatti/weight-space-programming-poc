# Evaluation: `runs/_baselines/reins/naive_L0-1-3-4-5/model`

**Test accuracy 0.7594 | macro-F1 0.6787 | weighted-F1 0.7582 | n=320**
Practice accuracy 0.8210 | macro-F1 0.8115 | gap +0.0616 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.600 | 0.692 | 0.643 | 26 |
| risk | 0.574 | 0.562 | 0.568 | 48 |
| strategy | 0.836 | 0.853 | 0.844 | 197 |
| metrics | 0.714 | 0.612 | 0.659 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 18 | 3 | 4 | 1 |
| 1 risk | 6 | 27 | 15 | 0 |
| 2 strategy | 5 | 13 | 168 | 11 |
| 3 metrics | 1 | 4 | 14 | 30 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 75,214,084 | 36,033,028 | 5 | 3072 | 300.9 MB | 8.6% | 16.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 21.078 ms/paragraph, batch 16: 25.306 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (75.94%: MISS); macro-F1 >= 0.6896 (0.6787: MISS)

# Evaluation: `runs/_baselines/reins/naive_L0-2-3-4-5/model`

**Test accuracy 0.7656 | macro-F1 0.6918 | weighted-F1 0.7693 | n=320**
Practice accuracy 0.8430 | macro-F1 0.8327 | gap +0.0774 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.600 | 0.692 | 0.643 | 26 |
| risk | 0.577 | 0.625 | 0.600 | 48 |
| strategy | 0.876 | 0.827 | 0.851 | 197 |
| metrics | 0.654 | 0.694 | 0.673 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 18 | 5 | 2 | 1 |
| 1 risk | 5 | 30 | 11 | 2 |
| 2 strategy | 6 | 13 | 163 | 15 |
| 3 metrics | 1 | 4 | 10 | 34 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 75,214,084 | 36,033,028 | 5 | 3072 | 300.9 MB | 8.6% | 16.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 21.026 ms/paragraph, batch 16: 25.215 ms/paragraph

Target: **PASS** — accuracy >= 76.50% (76.56%: ok); macro-F1 >= 0.6896 (0.6918: ok)

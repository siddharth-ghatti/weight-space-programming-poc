# Evaluation: `runs/_baselines/reins/ft_task_L4_e10/model`

**Test accuracy 0.7438 | macro-F1 0.6795 | weighted-F1 0.7459 | n=320**
Practice accuracy 0.9980 | macro-F1 0.9980 | gap +0.2542 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.692 | 0.692 | 0.692 | 26 |
| risk | 0.615 | 0.500 | 0.552 | 48 |
| strategy | 0.853 | 0.797 | 0.824 | 197 |
| metrics | 0.549 | 0.796 | 0.650 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 18 | 3 | 3 | 2 |
| 1 risk | 5 | 24 | 17 | 2 |
| 2 strategy | 3 | 9 | 157 | 28 |
| 3 metrics | 0 | 3 | 7 | 39 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.839 ms/paragraph, batch 16: 18.626 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (74.38%: MISS); macro-F1 >= 0.6896 (0.6795: MISS)

# Evaluation: `runs/_baselines/reins/ft_task_L6_F1536_e10/model`

**Test accuracy 0.7500 | macro-F1 0.6787 | weighted-F1 0.7506 | n=320**
Practice accuracy 0.9990 | macro-F1 0.9990 | gap +0.2490 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.708 | 0.654 | 0.680 | 26 |
| risk | 0.605 | 0.479 | 0.535 | 48 |
| strategy | 0.856 | 0.812 | 0.833 | 197 |
| metrics | 0.563 | 0.816 | 0.667 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 17 | 4 | 4 | 1 |
| 1 risk | 5 | 23 | 16 | 4 |
| 2 strategy | 2 | 9 | 160 | 26 |
| 3 metrics | 0 | 2 | 7 | 40 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,136,964 | 28,955,908 | 6 | 1536 | 272.6 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 17.148 ms/paragraph, batch 16: 19.709 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (75.00%: MISS); macro-F1 >= 0.6896 (0.6787: MISS)

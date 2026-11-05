# Evaluation: `runs/_baselines/reins/ft_task_L6_F1536/model`

**Test accuracy 0.7562 | macro-F1 0.6790 | weighted-F1 0.7573 | n=320**
Practice accuracy 0.9220 | macro-F1 0.9172 | gap +0.1658 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.615 | 0.615 | 0.615 | 26 |
| risk | 0.625 | 0.521 | 0.568 | 48 |
| strategy | 0.865 | 0.812 | 0.838 | 197 |
| metrics | 0.594 | 0.837 | 0.695 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 16 | 5 | 4 | 1 |
| 1 risk | 5 | 25 | 15 | 3 |
| 2 strategy | 5 | 8 | 160 | 24 |
| 3 metrics | 0 | 2 | 6 | 41 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,136,964 | 28,955,908 | 6 | 1536 | 272.6 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 17.162 ms/paragraph, batch 16: 19.674 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (75.62%: MISS); macro-F1 >= 0.6896 (0.6790: MISS)

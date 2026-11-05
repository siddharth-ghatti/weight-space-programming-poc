# Evaluation: `runs/_baselines/reins/ft_task_L3_e10/model`

**Test accuracy 0.7344 | macro-F1 0.6640 | weighted-F1 0.7375 | n=320**
Practice accuracy 0.9980 | macro-F1 0.9980 | gap +0.2636 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.727 | 0.615 | 0.667 | 26 |
| risk | 0.548 | 0.479 | 0.511 | 48 |
| strategy | 0.861 | 0.787 | 0.822 | 197 |
| metrics | 0.539 | 0.837 | 0.656 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 16 | 5 | 3 | 2 |
| 1 risk | 3 | 23 | 16 | 6 |
| 2 strategy | 3 | 12 | 155 | 27 |
| 3 metrics | 0 | 2 | 6 | 41 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 61,038,340 | 21,857,284 | 3 | 3072 | 244.2 MB | 25.8% | 49.3% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 12.785 ms/paragraph, batch 16: 13.731 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (73.44%: MISS); macro-F1 >= 0.6896 (0.6640: MISS)

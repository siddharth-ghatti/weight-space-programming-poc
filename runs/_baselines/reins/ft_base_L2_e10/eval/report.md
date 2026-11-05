# Evaluation: `runs/_baselines/reins/ft_base_L2_e10/model`

**Test accuracy 0.7219 | macro-F1 0.6647 | weighted-F1 0.7282 | n=320**
Practice accuracy 0.9760 | macro-F1 0.9607 | gap +0.2541 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.633 | 0.731 | 0.679 | 26 |
| risk | 0.543 | 0.521 | 0.532 | 48 |
| strategy | 0.870 | 0.746 | 0.803 | 197 |
| metrics | 0.533 | 0.816 | 0.645 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 19 | 5 | 0 | 2 |
| 1 risk | 5 | 25 | 15 | 3 |
| 2 strategy | 6 | 14 | 147 | 30 |
| 3 metrics | 0 | 2 | 7 | 40 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 53,950,468 | 14,769,412 | 2 | 3072 | 215.8 MB | 34.4% | 65.8% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 8.762 ms/paragraph, batch 16: 9.569 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (72.19%: MISS); macro-F1 >= 0.6896 (0.6647: MISS)

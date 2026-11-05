# Evaluation: `runs/_baselines/reins/naive_L0-2-3-5/model`

**Test accuracy 0.7719 | macro-F1 0.6851 | weighted-F1 0.7737 | n=320**
Practice accuracy 0.8060 | macro-F1 0.7932 | gap +0.0341 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.577 | 0.577 | 0.577 | 26 |
| risk | 0.564 | 0.646 | 0.602 | 48 |
| strategy | 0.866 | 0.853 | 0.859 | 197 |
| metrics | 0.733 | 0.673 | 0.702 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 15 | 6 | 4 | 1 |
| 1 risk | 5 | 31 | 11 | 1 |
| 2 strategy | 5 | 14 | 168 | 10 |
| 3 metrics | 1 | 4 | 11 | 33 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.979 ms/paragraph, batch 16: 20.264 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (77.19%: ok); macro-F1 >= 0.6896 (0.6851: MISS)

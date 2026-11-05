# Evaluation: `runs/_baselines/reins/naive_L1-2-3-5/model`

**Test accuracy 0.6312 | macro-F1 0.2709 | weighted-F1 0.5001 | n=320**
Practice accuracy 0.5250 | macro-F1 0.2119 | gap -0.1062 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.833 | 0.192 | 0.312 | 26 |
| risk | 0.000 | 0.000 | 0.000 | 48 |
| strategy | 0.627 | 1.000 | 0.771 | 197 |
| metrics | 0.000 | 0.000 | 0.000 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 5 | 0 | 21 | 0 |
| 1 risk | 1 | 0 | 47 | 0 |
| 2 strategy | 0 | 0 | 197 | 0 |
| 3 metrics | 0 | 0 | 49 | 0 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 68,126,212 | 28,945,156 | 4 | 3072 | 272.5 MB | 17.2% | 32.9% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 16.948 ms/paragraph, batch 16: 20.192 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (63.12%: MISS); macro-F1 >= 0.6896 (0.2709: MISS)

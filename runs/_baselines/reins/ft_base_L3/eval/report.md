# Evaluation: `runs/_baselines/reins/ft_base_L3/model`

**Test accuracy 0.6469 | macro-F1 0.3899 | weighted-F1 0.5954 | n=320**
Practice accuracy 0.6580 | macro-F1 0.4110 | gap +0.0111 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.000 | 0.000 | 0.000 | 26 |
| risk | 0.318 | 0.146 | 0.200 | 48 |
| strategy | 0.706 | 0.853 | 0.772 | 197 |
| metrics | 0.533 | 0.653 | 0.587 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 0 | 10 | 16 | 0 |
| 1 risk | 0 | 7 | 39 | 2 |
| 2 strategy | 0 | 3 | 168 | 26 |
| 3 metrics | 0 | 2 | 15 | 32 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 61,038,340 | 21,857,284 | 3 | 3072 | 244.2 MB | 25.8% | 49.3% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 12.493 ms/paragraph, batch 16: 14.056 ms/paragraph

Target: **FAIL** — accuracy >= 76.50% (64.69%: MISS); macro-F1 >= 0.6896 (0.3899: MISS)

# Evaluation: `runs/_baselines/reins/naive_L0-1-2-3-5/model`

**Test accuracy 0.7875 | macro-F1 0.7023 | weighted-F1 0.7814 | n=320**
Practice accuracy 0.8890 | macro-F1 0.8818 | gap +0.1015 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.696 | 0.615 | 0.653 | 26 |
| risk | 0.686 | 0.500 | 0.578 | 48 |
| strategy | 0.838 | 0.893 | 0.865 | 197 |
| metrics | 0.692 | 0.735 | 0.713 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 16 | 3 | 6 | 1 |
| 1 risk | 4 | 24 | 18 | 2 |
| 2 strategy | 3 | 5 | 176 | 13 |
| 3 metrics | 0 | 3 | 10 | 36 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 75,214,084 | 36,033,028 | 5 | 3072 | 300.9 MB | 8.6% | 16.4% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 20.995 ms/paragraph, batch 16: 25.316 ms/paragraph

Target: **PASS** — accuracy >= 76.50% (78.75%: ok); macro-F1 >= 0.6896 (0.7023: ok)

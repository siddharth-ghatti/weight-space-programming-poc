# Evaluation: `runs/r1/smoke/model`

**Test accuracy 0.7750 | macro-F1 0.7096 | weighted-F1 0.7749 | n=320**
Practice accuracy 0.9250 | macro-F1 0.9136 | gap +0.1500 (practice = the start model's own training data)

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| governance | 0.679 | 0.731 | 0.704 | 26 |
| risk | 0.643 | 0.562 | 0.600 | 48 |
| strategy | 0.856 | 0.843 | 0.849 | 197 |
| metrics | 0.643 | 0.735 | 0.686 | 49 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 governance | 19 | 2 | 4 | 1 |
| 1 risk | 5 | 27 | 14 | 2 |
| 2 strategy | 4 | 10 | 166 | 17 |
| 3 metrics | 0 | 3 | 10 | 36 |

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 82,301,956 | non-embedding reduction |
|---|---|---|---|---|---|---|
| 82,301,956 | 43,120,900 | 6 | 3072 | 329.2 MB | 0.0% | 0.0% |

CPU latency (4 threads, arm, max_len 512, median of 7): batch 1: 24.974 ms/paragraph, batch 16: 27.881 ms/paragraph

Target: **PASS** — accuracy >= 76.50% (77.50%: ok); macro-F1 >= 0.6896 (0.7096: ok)

## Editor's final answer

(auto-submitted at max turns)

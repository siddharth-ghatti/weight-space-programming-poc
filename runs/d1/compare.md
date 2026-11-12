# Distillation results (d1)

Insurance benchmark: 1,496 practice sentences (1,456 train the teacher and the edit, 40 kept back for doses and model selection), 1,779 held-out test sentences never used for anything. Reader: Qwen3.5-0.8B, frozen. "generated" reads the first word the detached model writes; "scored" takes the argmax over the eight answer words at the answer position.

| arm | test acc (generated) | macro-F1 (generated) | test acc (scored) | ECE | selective acc @80% | agrees with teacher | WikiText-2 ppl (before -> after) | ARC-Easy | capitals | contaminated | weights changed | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_s0 | 89.94% | 0.9003 | 89.60% | 0.010 | 96.70% | 89.49% | 28.75 -> 29.98 | 64.00% | 100.00% | 6.67% | 1,048,357 | 6142 |
| direct_s1 | 89.83% | 0.8989 | 89.54% | 0.009 | 96.63% | 89.49% | 28.75 -> 29.99 | 57.50% | 100.00% | 6.67% | 1,048,339 | 6267 |
| direct_s2 | 90.56% | 0.9068 | 90.95% | 0.021 | 97.82% | 91.06% | 28.75 -> 31.26 | 65.50% | 100.00% | 6.67% | 1,048,412 | 6389 |
| lora_s0 | 96.40% | 0.9642 | 96.40% | 0.009 | 99.86% | 95.78% | 28.75 -> 30.98 | 82.50% | 100.00% | 13.33% | 264,241,017 | 5342 |
| lora_s1 | 96.29% | 0.9631 | 96.29% | 0.019 | 99.72% | 95.90% | 28.75 -> 29.67 | 44.50% | 100.00% | 13.33% | 264,241,034 | 4746 |
| lora_s2 | 94.72% | 0.9483 | 94.66% | 0.018 | 99.30% | 94.72% | 28.75 -> 30.03 | 51.00% | 100.00% | 20.00% | 264,241,025 | 5568 |
| teacher_s0 | - | - | 98.82% | 0.006 | 99.93% | 100.00% | - | - | - | - | - | 627 |
| teacher_s1 | - | - | 98.76% | 0.006 | 99.93% | 100.00% | - | - | - | - | - | 626 |
| teacher_s2 | - | - | 98.76% | 0.006 | 99.93% | 100.00% | - | - | - | - | - | 626 |
| unedited | 58.07% | 0.5901 | 57.95% | 0.150 | 69.08% | 58.01% | 28.75 -> 28.75 | 68.50% | 100.00% | 0.00% | - | - |
| unedited_full_prompt | 61.21% | 0.6521 | 61.16% | 0.157 | 68.03% | 61.33% | - | - | - | - | - | 28 |
| ablation: full_method | - | - | 89.71% | 0.011 | 96.70% | 89.60% | 28.75 -> 29.98 | 64.00% | 100.00% | 6.67% | 1,048,576 | 1250 |
| ablation: hard_labels | - | - | 94.77% | 0.165 | 98.52% | 94.77% | 28.75 -> 29.25 | 66.50% | 100.00% | 6.67% | 1,048,576 | 1119 |
| ablation: no_protection | - | - | 57.95% | 0.150 | 69.08% | 58.01% | 28.75 -> 28.75 | 68.50% | 100.00% | 0.00% | - | 908 |
| ablation: random_change_same_size | - | - | 54.69% | 0.197 | 65.14% | 54.58% | 28.75 -> 31.58 | 63.00% | 100.00% | 6.67% | 1,048,576 | 0 |
| ablation: random_columns | - | - | 88.87% | 0.019 | 95.36% | 88.65% | 28.75 -> 30.46 | 63.50% | 100.00% | 0.00% | 1,048,576 | 758 |
| ablation: rows_25pct | - | - | 84.32% | 0.023 | 92.62% | 84.32% | 28.75 -> 29.31 | 65.50% | 100.00% | 0.00% | 1,048,576 | 343 |
| ablation: rows_50pct | - | - | 85.05% | 0.025 | 92.69% | 84.88% | 28.75 -> 29.26 | 66.00% | 100.00% | 0.00% | 1,048,576 | 477 |

Existing rows on the same test set (different base model: BERT-base, 110M):

| model | test acc | macro-F1 | API cost |
|---|---|---|---|
| BERT fine-tune (3 ep, local GPU) | 97.25% | 0.9727 | $0 |
| BERT linear probe | 88.53% | 0.8870 | $0 |
| BERT unedited | 13.10% | 0.0290 | $0 |
| Opus 5.5 edits BERT weights (v1/p50, 50 passes) | 98.31% | 0.9833 | $4.12 |

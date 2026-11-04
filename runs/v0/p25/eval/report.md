# Evaluation: `<repo>/runs/p25/model`

**Test accuracy 0.9843 | macro-F1 0.9845 | weighted-F1 0.9843 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0157

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.974 | 0.961 | 0.968 | 233 |
| exclusion | 0.965 | 0.982 | 0.974 | 226 |
| premium_payment | 0.973 | 0.995 | 0.984 | 219 |
| deductible_limits | 0.995 | 0.982 | 0.989 | 225 |
| claims_procedure | 0.995 | 0.991 | 0.993 | 222 |
| definitions | 1.000 | 0.995 | 0.998 | 206 |
| term_cancellation_renewal | 0.986 | 0.982 | 0.984 | 223 |
| conditions_duties | 0.987 | 0.987 | 0.987 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 224 | 6 | 0 | 1 | 0 | 0 | 2 | 0 |
| 1 exclusion | 1 | 222 | 0 | 0 | 0 | 0 | 1 | 2 |
| 2 premium_payment | 1 | 0 | 218 | 0 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 2 | 1 | 1 | 221 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 0 | 0 | 1 | 0 | 220 | 0 | 0 | 1 |
| 5 definitions | 0 | 1 | 0 | 0 | 0 | 205 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 3 | 0 | 0 | 0 | 219 | 0 |
| 7 conditions_duties | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 222 |

## Edited tensors

18 tensors, 7,087,880 entries changed (6.4736% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.154 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.370 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 1.046 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 62.190 | 28.241 |
| bert.encoder.layer.11.attention.self.query.bias | 768 | 1.0000 | 6.072 | 6.072 |
| bert.encoder.layer.11.attention.self.query.weight | 589,824 | 1.0000 | 34.948 | 34.948 |
| bert.encoder.layer.11.attention.self.value.bias | 768 | 1.0000 | 0.561 | 0.561 |
| bert.encoder.layer.11.attention.self.value.weight | 589,824 | 1.0000 | 40.898 | 30.114 |
| bert.encoder.layer.11.intermediate.dense.bias | 3,072 | 1.0000 | 9.586 | 5.159 |
| bert.encoder.layer.11.intermediate.dense.weight | 2,359,296 | 1.0000 | 60.345 | 60.183 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 1.471 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 10.198 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 42.376 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,296 | 1.0000 | 8341.866 | 55.033 |
| bert.pooler.dense.bias | 768 | 1.0000 | 0.961 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 21.921 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 28.284 | 0.000 |

## Editor's final answer

I submitted v6. In cross-validation (5-fold, 4 seeds), it scored about 99.6% accuracy and macro-F1 on practice sentences it hadn't been fitted on. I checked the edited model with a forward pass on 499 practice sentences: it got all of them right. Its logits matched my offline calculation to within 1.4e-4, and the smallest winning margin was 4.5. I can't measure the held-out test score; the cross-validation figure is my best estimate of it.

Only the last encoder layer (layer 11), the pooler and the classifier head were changed. Layers 0–10 and the embeddings are the original bert-base-uncased weights. Every new weight was computed in closed form from activations returned by forward passes; there was no training.

**What I changed**
1. **Sentence summary (layer-11 attention):** I zeroed the query weights so every token attends equally to all tokens. The [CLS] position then receives the average of all token vectors, which I scaled by 2 and added to its own vector. I set that layer's LayerNorm scale to 1 and offset to 0. The result at [CLS] is a normalized mix of the original [CLS] vector and the sentence average. In early tests this mix separated the classes better than either on its own (about 97.5% vs. 94.5% for [CLS] alone).
2. **Classifier (layer-11 feed-forward block):** I rebuilt the 3072-unit feed-forward block as a kernel classifier. Each unit is tied to one example sentence, and its GELU fires according to how similar the input is to that example. The output weights come from a weighted ridge regression onto one-hot labels, solved in closed form (lam = 0.005, kernel width 1). This non-linear version beat the best linear classifier in cross-validation (about 98% vs. 97.4% on practice data alone).
3. **Routing to the logits:** the feed-forward output writes the 8 class scores, scaled up, into 8 directions where normal activations barely vary. The final LayerNorm (scale 1, offset 0) keeps them intact. The pooler reads those 8 directions with a small scale so tanh stays in its well-behaved range, and the classifier is 10 × identity on those 8 units with zero bias. Because each step preserves the ordering of the scores, the model's prediction is the kernel classifier's prediction.

**Data used**
- The 1496 practice sentences, at full weight, all used as example units.
- 2201 extra labeled sentences I wrote myself, at weight 0.4–0.5. They cover all eight labels, all the lines of business, and the four registers (policy wording, broker summaries, claims letters, FAQs). The last batch of 255 targeted the label pairs the model still confused. Before writing them I checked how the practice data labels those pairs, for example: notice of loss counts as claims procedure, and a short-rate refund on cancellation counts as cancellation.
- Only 1576 of my sentences fit as example units, chosen at random, because the block has 3072 slots. All 3697 sentences were used to fit the output weights. Random selection scored the same as a smarter selection method.

**How accuracy grew**
| Version | Cross-validated accuracy |
|---|---|
| Linear classifier on the mixed vector | ~97.5% |
| Kernel classifier, practice data only | ~98% |
| + my first batches of sentences | ~99.2% |
| + batch 5 (v5) | ~99.5% |
| + targeted boundary batch (v6) | ~99.6% |

The remaining cross-validation errors are borderline cases, such as "give prompt notice of loss" (claims procedure vs. duties) and premium refunds tied to cancellation.

v5 (the same design without the last batch) was also verified exactly; my notes in `notes.md` have what's needed to restore it.

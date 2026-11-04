# Evaluation: `<repo>/runs/p50/model`

**Test accuracy 0.9730 | macro-F1 0.9734 | weighted-F1 0.9730 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0270

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.944 | 0.940 | 0.942 | 233 |
| exclusion | 0.952 | 0.956 | 0.954 | 226 |
| premium_payment | 0.951 | 0.982 | 0.966 | 219 |
| deductible_limits | 0.987 | 0.978 | 0.982 | 225 |
| claims_procedure | 0.995 | 0.986 | 0.991 | 222 |
| definitions | 0.990 | 1.000 | 0.995 | 206 |
| term_cancellation_renewal | 0.982 | 0.964 | 0.973 | 223 |
| conditions_duties | 0.987 | 0.982 | 0.984 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 219 | 10 | 1 | 0 | 0 | 0 | 2 | 1 |
| 1 exclusion | 6 | 216 | 0 | 0 | 0 | 2 | 2 | 0 |
| 2 premium_payment | 1 | 0 | 215 | 3 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 3 | 1 | 1 | 220 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 0 | 0 | 0 | 219 | 0 | 0 | 2 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 7 | 0 | 0 | 0 | 215 | 0 |
| 7 conditions_duties | 1 | 0 | 2 | 0 | 1 | 0 | 0 | 221 |

## Edited tensors

18 tensors, 7,087,880 entries changed (6.4736% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.154 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.370 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 1.046 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 27712.797 | 28.241 |
| bert.encoder.layer.11.attention.self.query.bias | 768 | 1.0000 | 6.072 | 6.072 |
| bert.encoder.layer.11.attention.self.query.weight | 589,824 | 1.0000 | 34.948 | 34.948 |
| bert.encoder.layer.11.attention.self.value.bias | 768 | 1.0000 | 0.561 | 0.561 |
| bert.encoder.layer.11.attention.self.value.weight | 589,824 | 1.0000 | 40.898 | 30.114 |
| bert.encoder.layer.11.intermediate.dense.bias | 3,072 | 1.0000 | 11.026 | 5.159 |
| bert.encoder.layer.11.intermediate.dense.weight | 2,359,296 | 1.0000 | 60.468 | 60.183 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 1.471 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 10.198 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 789.162 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,296 | 1.0000 | 86702.789 | 55.033 |
| bert.pooler.dense.bias | 768 | 1.0000 | 0.961 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 21.921 | 21.921 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 8 | 0.0013 | 2.828 | 0.000 |

## Editor's final answer

I rewrote the last encoder layer, the pooler and the classifier head; layers 0–10 are unchanged. Cross-validation on the practice set estimates about 98% accuracy (97.7% averaged over three different fold splits). On a held-out fold that the model never used, the real edited model scored 98.4% accuracy and 98.4% macro-F1, exactly matching the closed-form prediction. It also got all 16 sentences I wrote myself correct. No training was used: every weight comes from closed-form math on hidden states read through the forward-pass tool.

**How I chose the features.** I read the frozen BERT's hidden states for all 1,496 practice sentences and tested a simple linear classifier on each layer's output. Averaging the input to the last layer over all words beat the [CLS] vector: 97.2% vs 94.1%. A similarity-based (kernel) classifier on that average raised it further, to about 98%. This helps because the data contains many reworded versions of the same clause.

**What I changed:**
1. **Averaging (layer 11 attention).** Query weights are set to zero, so [CLS] attends equally to every real token. Value = identity and output = 1000 × identity, which makes the original [CLS] vector negligible. Both LayerNorms are set to weight 1, bias 0. Reading the edited model back confirmed the final [CLS] equals the normalised average (cosine 0.9999997).
2. **Classifier (layer 11 FFN).** 1,496 GELU units each fire on similarity to one practice sentence (scale 2, offset 0, after standardising each dimension). Sixteen more units carry a linear term exactly, using GELU(x) − GELU(−x) = x. The output weights come from a single regularised least-squares solve (strength 0.5) against one-hot labels. They write the 8 class scores, ×10⁴, into 8 fixed directions in the hidden vector, so the final LayerNorm only rescales them and the winning class is preserved.
3. **Pooler and classifier.** Pooler rows 0–7 read those 8 directions at scale 0.05, which keeps tanh unsaturated and in the same order. The classifier is an identity on the first 8 pooled outputs, so the predicted label is the class with the highest score.

**What didn't help:**
- Random tanh features in the pooler did worse (94.0%).
- Multi-scale or shifted similarity units and PCA whitening all stayed at about 97.7%.

**Remaining errors** are mostly genuinely ambiguous pairs: notice duties vs claims procedure, cancellation refunds vs premium, and limits vs coverage.

**One gain I left out:** combining averages from several layers looked worth about +0.4% in idealised cross-validation. I didn't build it, because it would need edits that only affect [CLS] early in the network and would then have to survive about ten layers.

All 18 edited tensors are saved (copies in `ckpt_B/`), the builder code is in `build_ffn.py`, and the full log is in `notes.md`.

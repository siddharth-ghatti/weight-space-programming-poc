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
# Evaluation: `<repo>/runs/p10/model`

**Test accuracy 0.9668 | macro-F1 0.9672 | weighted-F1 0.9669 | n=1779**
Practice accuracy 0.9967 | macro-F1 0.9967 | gap +0.0298

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.953 | 0.953 | 0.953 | 233 |
| exclusion | 0.964 | 0.960 | 0.962 | 226 |
| premium_payment | 0.930 | 0.977 | 0.953 | 219 |
| deductible_limits | 0.986 | 0.969 | 0.978 | 225 |
| claims_procedure | 0.982 | 0.973 | 0.977 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.977 | 0.946 | 0.961 | 223 |
| conditions_duties | 0.952 | 0.960 | 0.956 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 222 | 6 | 0 | 1 | 0 | 0 | 3 | 1 |
| 1 exclusion | 3 | 217 | 0 | 0 | 0 | 1 | 1 | 4 |
| 2 premium_payment | 1 | 0 | 214 | 2 | 0 | 0 | 0 | 2 |
| 3 deductible_limits | 3 | 2 | 2 | 218 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 0 | 1 | 0 | 216 | 0 | 0 | 4 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 11 | 0 | 0 | 0 | 211 | 0 |
| 7 conditions_duties | 2 | 0 | 2 | 0 | 4 | 0 | 1 | 216 |

## Edited tensors

40 tensors, 13,592,840 entries changed (12.4149% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.10.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.648 | 3.648 |
| bert.encoder.layer.10.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.731 | 23.585 |
| bert.encoder.layer.10.attention.output.dense.bias | 768 | 1.0000 | 1.460 | 1.460 |
| bert.encoder.layer.10.attention.output.dense.weight | 589,824 | 1.0000 | 25.615 | 25.615 |
| bert.encoder.layer.10.output.LayerNorm.bias | 768 | 1.0000 | 2.194 | 2.194 |
| bert.encoder.layer.10.output.LayerNorm.weight | 768 | 1.0000 | 5.345 | 22.714 |
| bert.encoder.layer.10.output.dense.bias | 768 | 1.0000 | 2.841 | 2.841 |
| bert.encoder.layer.10.output.dense.weight | 2,359,296 | 1.0000 | 58.249 | 58.249 |
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.154 | 3.154 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 768 | 1.0000 | 4.370 | 23.686 |
| bert.encoder.layer.11.attention.output.dense.bias | 768 | 1.0000 | 1.046 | 1.046 |
| bert.encoder.layer.11.attention.output.dense.weight | 589,824 | 1.0000 | 28.241 | 28.241 |
| bert.encoder.layer.11.output.LayerNorm.bias | 768 | 1.0000 | 1.471 | 1.471 |
| bert.encoder.layer.11.output.LayerNorm.weight | 768 | 1.0000 | 10.198 | 17.560 |
| bert.encoder.layer.11.output.dense.bias | 768 | 1.0000 | 1.590 | 1.590 |
| bert.encoder.layer.11.output.dense.weight | 2,359,296 | 1.0000 | 55.033 | 55.033 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.613 | 3.613 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.350 | 23.364 |
| bert.encoder.layer.8.attention.output.dense.bias | 768 | 1.0000 | 1.769 | 1.769 |
| bert.encoder.layer.8.attention.output.dense.weight | 589,824 | 1.0000 | 27712.795 | 25.772 |
| bert.encoder.layer.8.attention.self.query.bias | 768 | 1.0000 | 4.838 | 4.838 |
| bert.encoder.layer.8.attention.self.query.weight | 589,824 | 1.0000 | 33.808 | 33.808 |
| bert.encoder.layer.8.attention.self.value.bias | 768 | 1.0000 | 1.227 | 1.227 |
| bert.encoder.layer.8.attention.self.value.weight | 589,824 | 1.0000 | 38.854 | 27.192 |
| bert.encoder.layer.8.output.LayerNorm.bias | 768 | 1.0000 | 1.962 | 1.962 |
| bert.encoder.layer.8.output.LayerNorm.weight | 768 | 1.0000 | 4.826 | 23.060 |
| bert.encoder.layer.8.output.dense.bias | 768 | 1.0000 | 2.401 | 2.401 |
| bert.encoder.layer.8.output.dense.weight | 2,359,296 | 1.0000 | 57.365 | 57.365 |
| bert.encoder.layer.9.attention.output.LayerNorm.bias | 768 | 1.0000 | 3.780 | 3.780 |
| bert.encoder.layer.9.attention.output.LayerNorm.weight | 768 | 1.0000 | 5.537 | 22.757 |
| bert.encoder.layer.9.attention.output.dense.bias | 768 | 1.0000 | 1.703 | 1.703 |
| bert.encoder.layer.9.attention.output.dense.weight | 589,824 | 1.0000 | 25.113 | 25.113 |
| bert.encoder.layer.9.output.LayerNorm.bias | 768 | 1.0000 | 1.811 | 1.811 |
| bert.encoder.layer.9.output.LayerNorm.weight | 768 | 1.0000 | 5.776 | 22.247 |
| bert.encoder.layer.9.output.dense.bias | 768 | 1.0000 | 2.219 | 2.219 |
| bert.encoder.layer.9.output.dense.weight | 2,359,296 | 1.0000 | 59.883 | 59.883 |
| bert.pooler.dense.bias | 768 | 1.0000 | 3.710 | 0.961 |
| bert.pooler.dense.weight | 589,824 | 1.0000 | 22.805 | 21.921 |
| classifier.bias | 8 | 1.0000 | 5.254 | 0.000 |
| classifier.weight | 6,144 | 1.0000 | 75.578 | 0.000 |

## Editor's final answer

I turned bert-base-uncased into an 8-label insurance classifier using closed-form math on real activations, with no training. Ten-fold cross-validation on the practice set estimates about 97% held-out accuracy (5-fold: 96.6–96.8%). A check pass showed the edited model's logits match my offline calculation to within 5e-5.

**Why this design.** I read the base model's hidden states for 1,024 practice sentences and compared closed-form probes. The [CLS] token was a weaker sentence feature (about 93% at the last layer). The plain average of all token states from the middle-to-late layers (8–10) was best, at about 96%. Layer 8 came out marginally on top, so I rebuilt the network so that [CLS] ends up holding that average.

**What I changed:**
1. **Layer 8 becomes a pooling layer.**
   - Query weights and bias are set to zero, so every token attends equally to all real tokens (padding stays masked).
   - Value is set to the identity and the attention output to 1000 × identity. The averaged vector then swamps [CLS]'s own state.
   - Both LayerNorms are set to weight 1 and bias 0, and the feed-forward output is set to zero.
   - The result is that [CLS] becomes the normalised average of the layer-8 token states.
2. **Layers 9–11 pass [CLS] through unchanged.** Their attention and feed-forward outputs are zero, and their LayerNorms are set to weight 1 and bias 0.
3. **The pooler is a per-dimension standardiser with a small scale (0.1)**, so the tanh stays nearly linear. Its centring and scaling come from activation statistics.
4. **The classifier head was fitted in closed form.** I read the edited model's actual pooled outputs for all 1,496 practice sentences. Then I fitted an equal-weight average of a ridge regression (one-hot targets, λ=200) and a shrinkage linear discriminant (shrinkage 0.2). Both were fitted on standardised features, with the standardisation folded into `classifier.weight` and `classifier.bias`. The regularisation was chosen by repeated 5-fold cross-validation.

**Checks:**
- The edited [CLS] output matched the normalised layer-8 average to within 0.015.
- On 512 practice items the model scores 99.4%, but that is optimistic because those items were used for fitting.
- The remaining errors are spread across classes and registers, and are mostly borderline cases such as "is a covered loss" read as an exclusion.

**What I tried and didn't use.** Combining the layer-0 word-embedding average with a deep layer scored about 1% higher in cross-validation. But it needs about 256+ dimensions of extra information, and the only place to carry it is [CLS]. Other tokens use [CLS] as an attention sink in layers 2–4, so changing it there would distort them. Routing the layer-9 [CLS] state alongside the average gained only about 0.15%. Mixing two streams through the residual connection, or adding sentence length, gained nothing. I judged these not worth the passes or the risk.

**Budget.** I used 6 of the 10 forward passes; one was wasted early by sending a single id instead of 1,024. A backup of the final edit set is in `edits_v1_backup/`, and running notes are in `notes.md`.

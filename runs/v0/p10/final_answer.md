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
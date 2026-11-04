**Result:** a 10-layer model with **95,312,648 parameters**, 12.9% smaller than the original 109,488,392. Its honest 5-fold cross-validated accuracy on the practice set is **about 98.1%**, against 97.8% for the start model's own readout (which scored 98.31% on test). I estimate roughly a 1–4% chance it falls below the 97.8% bar on the test set.

**Structure:** I keep original layers 0–8 unchanged, plus original layer 11, which I rewrote as a pooling layer. All FFN blocks stay at full width (3072 units).

**Why this size and no smaller**
- **Narrowing the FFN blocks didn't work.** Pruning units, even with their average output folded into the bias, damaged the deep features badly. At 1536 units accuracy dropped about 0.75%. At 2560 units the 9-layer features scored 97.36%, worse than simply dropping a layer (97.6%).
- **Accuracy tracks the number of real layers kept.** Skipping layers in the middle saved nothing beyond what dropping the same number from the end would.
- **The last kept layer only matters at the [CLS] position.** So it can't usefully be a normal layer; it is better used as a pooling layer.
- **9 real layers vs 8.**
  - With 9 real layers (k=10), my best buildable readout reaches about 98.05–98.1% CV.
  - With 8 real layers (k=9, 88.2M), the same readout reaches about 97.65–97.7%.
  - Compared item by item against the start model, the 8-layer version had roughly a 14–19% chance of failing the 97.8% bar, against about 1–3% for 9 layers. I judged a 7.5% size saving not worth that risk.

**How the model works (all closed-form, nothing trained iteratively)**
1. **Carry heads.** The start model's carry heads in layers 2 and 6 already write 8 class scores, computed from the average token vector, into 8 reserved hidden dimensions. I re-solved their value weights so they write the linear part of my own model, from the layer-2 and layer-6 averages. I measured each sentence's carry scale from real activations, and the reads matched that description to about 1e-5.
2. **Pooling layer (original layer 11).**
   - Its attention has all queries and keys at zero, so every head attends evenly to all tokens. Values are 8×identity and the output weights are identity, so [CLS] becomes LayerNorm(own state + 8 × average of the layer-8 output tokens).
   - Its FFN is an exact kernel machine. 1,496 units each hold one practice sentence as a center and compute GELU of a fixed inner product between the pooled vector and that center. Their output weights are the kernel-ridge coefficients (penalty 0.03), written as class scores into the 8 reserved dimensions. The other 1,576 units are inert.
3. **Pooler and classifier.** Pooler rows 0–7 read the 8 reserved dimensions after subtracting their shared mean, with a gain of 10. The classifier picks pooled dims 0–7. Since tanh preserves order, the prediction is the argmax of the kernel-ridge score.

**How I fitted it:** I read the pooled [CLS] vectors from the real network (practice sentences only) and computed the kernel matrix from them. I then solved one joint kernel ridge combining the GELU kernel on the pooled vector with the carry-scaled linear terms on the layer-2 and layer-6 averages. The 98.1% figure is cross-validation of exactly this model.

**Checks:** Forward passes over all 1496 practice sentences gave logits equal to my numpy prediction (tanh of the centered ridge score) within 7e-6. In-sample accuracy was 100% with a minimum top-1 logit margin of 0.77; that only shows the edits work as designed, not how it will do on test.
# Evaluation: `runs/c1/p25/model`

**Test accuracy 0.9826 | macro-F1 0.9828 | weighted-F1 0.9826 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0174

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.970 | 0.966 | 0.968 | 233 |
| exclusion | 0.965 | 0.978 | 0.971 | 226 |
| premium_payment | 0.964 | 0.991 | 0.977 | 219 |
| deductible_limits | 0.995 | 0.978 | 0.987 | 225 |
| claims_procedure | 1.000 | 0.995 | 0.998 | 222 |
| definitions | 0.995 | 0.995 | 0.995 | 206 |
| term_cancellation_renewal | 0.982 | 0.964 | 0.973 | 223 |
| conditions_duties | 0.991 | 0.996 | 0.993 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 225 | 5 | 0 | 0 | 0 | 0 | 2 | 1 |
| 1 exclusion | 2 | 221 | 0 | 0 | 0 | 1 | 1 | 1 |
| 2 premium_payment | 1 | 0 | 217 | 1 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 2 | 2 | 1 | 220 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 1 | 0 | 0 | 0 | 221 | 0 | 0 | 0 |
| 5 definitions | 0 | 1 | 0 | 0 | 0 | 205 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 7 | 0 | 0 | 0 | 215 | 0 |
| 7 conditions_duties | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 224 |

## Edited tensors

24 tensors, 2,984,792 entries changed (2.7261% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.attention.output.dense.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.output.dense.weight | 768 | 0.0013 | 27.713 | 55.426 |
| bert.encoder.layer.11.attention.self.key.bias | 768 | 1.0000 | 0.146 | 0.146 |
| bert.encoder.layer.11.attention.self.key.weight | 589,824 | 1.0000 | 34.171 | 34.171 |
| bert.encoder.layer.11.attention.self.query.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.query.weight | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.value.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.value.weight | 768 | 0.0013 | 193.990 | 27.713 |
| bert.encoder.layer.11.intermediate.dense.bias | 3,072 | 1.0000 | 6.553 | 5.159 |
| bert.encoder.layer.11.intermediate.dense.weight | 2,359,296 | 1.0000 | 60.264 | 60.183 |
| bert.encoder.layer.11.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.output.dense.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.output.dense.weight | 11,968 | 0.0051 | 5.002 | 0.000 |
| bert.encoder.layer.2.attention.self.value.bias | 8 | 0.0104 | 0.007 | 1.280 |
| bert.encoder.layer.2.attention.self.value.weight | 6,080 | 0.0103 | 0.047 | 20.851 |
| bert.encoder.layer.6.attention.self.value.bias | 8 | 0.0104 | 0.008 | 0.859 |
| bert.encoder.layer.6.attention.self.value.weight | 6,080 | 0.0103 | 0.056 | 25.530 |
| bert.pooler.dense.bias | 8 | 0.0104 | 0.336 | 0.336 |
| bert.pooler.dense.weight | 6,144 | 0.0104 | 29.581 | 13.231 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 0 | 0.0000 | 0.000 | 2.828 |

## Editor's final answer

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

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 95,312,648 | 71,475,464 | 10 | 3072 | 381.3 MB | 13.0% |

CPU latency (4 threads, arm, median of 7): batch 1: 29.473 ms/sentence, batch 32: 12.493 ms/sentence

Accuracy target 97.8%: **PASS** (test accuracy 98.26%)

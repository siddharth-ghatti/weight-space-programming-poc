# Evaluation: `runs/c1/p50/model`

**Test accuracy 0.9792 | macro-F1 0.9795 | weighted-F1 0.9792 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0208

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.962 | 0.966 | 0.964 | 233 |
| exclusion | 0.978 | 0.978 | 0.978 | 226 |
| premium_payment | 0.968 | 0.977 | 0.973 | 219 |
| deductible_limits | 0.978 | 0.991 | 0.985 | 225 |
| claims_procedure | 0.991 | 0.977 | 0.984 | 222 |
| definitions | 0.995 | 1.000 | 0.998 | 206 |
| term_cancellation_renewal | 0.978 | 0.978 | 0.978 | 223 |
| conditions_duties | 0.986 | 0.969 | 0.978 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 225 | 3 | 1 | 1 | 0 | 0 | 1 | 2 |
| 1 exclusion | 3 | 221 | 0 | 0 | 0 | 1 | 1 | 0 |
| 2 premium_payment | 1 | 0 | 214 | 3 | 0 | 0 | 1 | 0 |
| 3 deductible_limits | 0 | 1 | 1 | 223 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 2 | 1 | 0 | 1 | 217 | 0 | 0 | 1 |
| 5 definitions | 0 | 0 | 0 | 0 | 0 | 206 | 0 | 0 |
| 6 term_cancellation_renewal | 0 | 0 | 5 | 0 | 0 | 0 | 218 | 0 |
| 7 conditions_duties | 3 | 0 | 0 | 0 | 2 | 0 | 2 | 218 |

## Edited tensors

94 tensors, 30,696,976 entries changed (28.0367% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.0.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.460 | 9.131 |
| bert.encoder.layer.0.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.134 | 26.663 |
| bert.encoder.layer.0.attention.output.dense.bias | 768 | 1.0000 | 0.321 | 0.983 |
| bert.encoder.layer.0.attention.output.dense.weight | 54,784 | 0.0929 | 9.562 | 21.947 |
| bert.encoder.layer.0.attention.self.query.bias | 64 | 0.0833 | 2.514 | 7.781 |
| bert.encoder.layer.0.attention.self.query.weight | 49,152 | 0.0833 | 8.304 | 33.116 |
| bert.encoder.layer.0.attention.self.value.bias | 64 | 0.0833 | 1.037 | 1.129 |
| bert.encoder.layer.0.attention.self.value.weight | 49,152 | 0.0833 | 15.274 | 22.127 |
| bert.encoder.layer.0.intermediate.dense.bias | 3,072 | 1.0000 | 10.798 | 6.588 |
| bert.encoder.layer.0.intermediate.dense.weight | 2,359,296 | 1.0000 | 18.153 | 57.479 |
| bert.encoder.layer.0.output.LayerNorm.bias | 8 | 0.0104 | 0.238 | 2.912 |
| bert.encoder.layer.0.output.LayerNorm.weight | 8 | 0.0104 | 0.626 | 21.005 |
| bert.encoder.layer.0.output.dense.bias | 768 | 1.0000 | 2.469 | 2.222 |
| bert.encoder.layer.0.output.dense.weight | 2,359,296 | 1.0000 | 45.841 | 55.062 |
| bert.encoder.layer.1.attention.output.LayerNorm.bias | 8 | 0.0104 | 0.370 | 5.384 |
| bert.encoder.layer.1.attention.output.LayerNorm.weight | 8 | 0.0104 | 0.325 | 24.390 |
| bert.encoder.layer.1.attention.output.dense.bias | 8 | 0.0104 | 0.119 | 1.673 |
| bert.encoder.layer.1.attention.output.dense.weight | 6,144 | 0.0104 | 1.920 | 21.504 |
| bert.encoder.layer.1.attention.self.key.bias | 768 | 1.0000 | 6.816 | 0.142 |
| bert.encoder.layer.1.attention.self.key.weight | 589,824 | 1.0000 | 8.889 | 32.659 |
| bert.encoder.layer.1.attention.self.query.bias | 768 | 1.0000 | 7.316 | 5.024 |
| bert.encoder.layer.1.attention.self.query.weight | 589,824 | 1.0000 | 9.279 | 32.666 |
| bert.encoder.layer.1.attention.self.value.bias | 768 | 1.0000 | 5.358 | 1.161 |
| bert.encoder.layer.1.attention.self.value.weight | 589,824 | 1.0000 | 6.748 | 21.783 |
| bert.encoder.layer.1.intermediate.dense.bias | 3,072 | 1.0000 | 13.873 | 6.174 |
| bert.encoder.layer.1.intermediate.dense.weight | 2,359,296 | 1.0000 | 18.763 | 60.250 |
| bert.encoder.layer.1.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 2.552 |
| bert.encoder.layer.1.output.LayerNorm.weight | 8 | 0.0104 | 2.828 | 24.038 |
| bert.encoder.layer.1.output.dense.bias | 768 | 1.0000 | 2.040 | 1.679 |
| bert.encoder.layer.1.output.dense.weight | 2,359,296 | 1.0000 | 48.339 | 58.159 |
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.attention.output.dense.bias | 760 | 0.9896 | 191.731 | 0.000 |
| bert.encoder.layer.11.attention.output.dense.weight | 768 | 0.0013 | 496.258 | 55.426 |
| bert.encoder.layer.11.attention.self.key.bias | 768 | 1.0000 | 0.146 | 0.146 |
| bert.encoder.layer.11.attention.self.key.weight | 589,824 | 1.0000 | 37.018 | 34.171 |
| bert.encoder.layer.11.attention.self.query.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.query.weight | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.value.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.value.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.intermediate.dense.bias | 3,072 | 1.0000 | 5.159 | 5.159 |
| bert.encoder.layer.11.intermediate.dense.weight | 2,359,296 | 1.0000 | 60.324 | 60.183 |
| bert.encoder.layer.11.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.output.dense.bias | 8 | 0.0104 | 0.260 | 0.000 |
| bert.encoder.layer.11.output.dense.weight | 8,080 | 0.0034 | 27.272 | 0.000 |
| bert.encoder.layer.2.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 4.881 |
| bert.encoder.layer.2.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 24.164 |
| bert.encoder.layer.2.attention.output.dense.weight | 8 | 0.0000 | 2.828 | 20.315 |
| bert.encoder.layer.2.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 2.231 |
| bert.encoder.layer.2.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.677 |
| bert.encoder.layer.2.output.dense.bias | 760 | 0.9896 | 2.347 | 1.976 |
| bert.encoder.layer.2.output.dense.weight | 2,334,720 | 0.9896 | 48.498 | 58.496 |
| bert.encoder.layer.3.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 4.425 |
| bert.encoder.layer.3.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 24.172 |
| bert.encoder.layer.3.output.LayerNorm.bias | 760 | 0.9896 | 0.772 | 1.708 |
| bert.encoder.layer.3.output.LayerNorm.weight | 760 | 0.9896 | 0.899 | 22.564 |
| bert.encoder.layer.3.output.dense.bias | 760 | 0.9896 | 3.721 | 2.025 |
| bert.encoder.layer.3.output.dense.weight | 2,334,720 | 0.9896 | 53.325 | 60.103 |
| bert.encoder.layer.4.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 4.017 |
| bert.encoder.layer.4.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.540 |
| bert.encoder.layer.4.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 1.645 |
| bert.encoder.layer.4.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.348 |
| bert.encoder.layer.4.output.dense.bias | 760 | 0.9896 | 1.860 | 1.490 |
| bert.encoder.layer.4.output.dense.weight | 2,334,720 | 0.9896 | 50.554 | 60.944 |
| bert.encoder.layer.5.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 3.809 |
| bert.encoder.layer.5.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.705 |
| bert.encoder.layer.5.output.LayerNorm.bias | 760 | 0.9896 | 0.868 | 1.865 |
| bert.encoder.layer.5.output.LayerNorm.weight | 760 | 0.9896 | 0.438 | 23.133 |
| bert.encoder.layer.5.output.dense.bias | 760 | 0.9896 | 6.620 | 1.599 |
| bert.encoder.layer.5.output.dense.weight | 2,334,720 | 0.9896 | 53.665 | 60.076 |
| bert.encoder.layer.6.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 3.936 |
| bert.encoder.layer.6.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.179 |
| bert.encoder.layer.6.attention.output.dense.weight | 8 | 0.0000 | 2.828 | 24.285 |
| bert.encoder.layer.6.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 1.980 |
| bert.encoder.layer.6.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.173 |
| bert.encoder.layer.6.output.dense.bias | 760 | 0.9896 | 2.301 | 1.938 |
| bert.encoder.layer.6.output.dense.weight | 2,334,720 | 0.9896 | 49.217 | 59.298 |
| bert.encoder.layer.7.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 4.035 |
| bert.encoder.layer.7.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.023 |
| bert.encoder.layer.7.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 2.011 |
| bert.encoder.layer.7.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 22.534 |
| bert.encoder.layer.7.output.dense.bias | 760 | 0.9896 | 2.367 | 2.255 |
| bert.encoder.layer.7.output.dense.weight | 2,334,720 | 0.9896 | 47.320 | 57.324 |
| bert.encoder.layer.8.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 3.602 |
| bert.encoder.layer.8.attention.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.413 |
| bert.encoder.layer.8.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 1.957 |
| bert.encoder.layer.8.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 23.112 |
| bert.encoder.layer.8.output.dense.bias | 760 | 0.9896 | 2.332 | 2.396 |
| bert.encoder.layer.8.output.dense.weight | 2,334,720 | 0.9896 | 47.055 | 57.089 |
| bert.pooler.dense.bias | 8 | 0.0104 | 0.336 | 0.336 |
| bert.pooler.dense.weight | 6,144 | 0.0104 | 14.357 | 13.231 |
| classifier.bias | 0 | 0.0000 | 0.000 | 0.000 |
| classifier.weight | 0 | 0.0000 | 0.000 | 2.828 |

## Editor's final answer

Final model: 8 of the 12 encoder layers (original layers 0, 1, 2, 3, 5, 7, 8 and 11), each with 1,024 of its 3,072 feed-forward units. That is 55,954,696 parameters, 48.9% fewer than the original 109,488,392. I estimate test accuracy at about 98.2–98.3%; the best cross-validation proxy is 98.28%, against the 97.8% pass mark. On the practice set the deployed model gets all 1,496 sentences right, but it was fit on them, so that only shows the edits work as designed.

What I changed, all with closed-form fits and no training:

1. **Linear class-evidence channel from the start.** Eight of the 768 hidden dimensions are reserved for class scores. In layer 0, one attention head averages the token embeddings and writes a linear class score from that average into those 8 dimensions. Every later layer is cut off from reading or writing them, so the scores reach the end untouched.

2. **Narrower content layers.** The content layers are original layers 0, 1, 2, 3, 5, 7 and 8. In each one I kept the 1,024 feed-forward units with the most output energy. Then I refit that block's output weights by ridge regression to match the original model's hidden states, one layer at a time and starting from the already-pruned layers below.

3. **Two dropped content layers.** Original layers 6 and 4 are gone. To cover for them, layers 5 and 3 had their feed-forward blocks refit to produce the hidden state the original model had two layers later.
   - On the same cross-validation folds, the final setup scores 98.28% without layer 4 and 98.24% with it, so dropping layer 4 cost nothing measurable.
   - I didn't try dropping more. A test of layer 8 showed its stand-in layer couldn't approximate it, and cutting the model short there had cost about 0.46% earlier.

4. **Layer 11 as the classifier.**
   - Its attention gives every token equal weight, so the first ([CLS]) position receives the average of the last content layer's token vectors.
   - Its 1,024 feed-forward units compute a kernel classifier on that average. 1,008 units are anchor points chosen from practice sentences, and 16 units pass the layer-0 scores through linearly.
   - The pooler reads only the 8 reserved dimensions and removes their shared offset. The classifier just picks those 8 outputs.

**A fix found in this final session.** The earlier way of fitting layer 11 kept an old, fixed layer-0 direction and refit only the 8×8 mixing weights on the same training data. Simulation showed this overweights the linear part and loses about 0.25% (97.79% vs 98.06% in 10-fold cross-validation). Instead, I fit one combined model (kernel plus linear part) on all practice sentences. I wrote its linear part directly into the layer-0 head, scaled to the old size so the downstream normalisation barely changes. I installed the kernel weights exactly as fitted. Simulated end to end, this matches the combined model's cross-validation.

**Checks.**
- I read logits for all 1,496 practice ids. They matched my numpy recomputation of layer 11 from the measured hidden states to within 5e-6.
- Accuracy on them is 100%, the smallest gap between the top two classes is 0.44, and nothing saturates (logits range from −0.46 to 0.93).
- Changing the layer-0 direction shifted the content features by only 0.5%.

Two forward passes were used for that verification, and 3 of the 50 are left unused.

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 55,954,696 | 32,117,512 | 8 | 1024 | 223.8 MB | 48.9% |

CPU latency (4 threads, arm, median of 7): batch 1: 15.838 ms/sentence, batch 32: 5.847 ms/sentence

Accuracy target 97.8%: **PASS** (test accuracy 97.92%)

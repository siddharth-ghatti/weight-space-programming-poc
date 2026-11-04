# Evaluation: `runs/c1/p10/model`

**Test accuracy 0.9741 | macro-F1 0.9745 | weighted-F1 0.9741 | n=1779**
Practice accuracy 1.0000 | macro-F1 1.0000 | gap +0.0259

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.961 | 0.944 | 0.952 | 233 |
| exclusion | 0.952 | 0.969 | 0.961 | 226 |
| premium_payment | 0.956 | 0.986 | 0.971 | 219 |
| deductible_limits | 0.991 | 0.978 | 0.984 | 225 |
| claims_procedure | 0.986 | 0.986 | 0.986 | 222 |
| definitions | 0.995 | 0.995 | 0.995 | 206 |
| term_cancellation_renewal | 0.982 | 0.960 | 0.971 | 223 |
| conditions_duties | 0.973 | 0.978 | 0.976 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 220 | 9 | 0 | 0 | 0 | 0 | 2 | 2 |
| 1 exclusion | 4 | 219 | 0 | 0 | 0 | 1 | 1 | 1 |
| 2 premium_payment | 1 | 0 | 216 | 2 | 0 | 0 | 0 | 0 |
| 3 deductible_limits | 2 | 1 | 2 | 220 | 0 | 0 | 0 | 0 |
| 4 claims_procedure | 0 | 0 | 0 | 0 | 219 | 0 | 0 | 3 |
| 5 definitions | 0 | 1 | 0 | 0 | 0 | 205 | 0 | 0 |
| 6 term_cancellation_renewal | 1 | 0 | 7 | 0 | 1 | 0 | 214 | 0 |
| 7 conditions_duties | 1 | 0 | 1 | 0 | 2 | 0 | 1 | 220 |

## Edited tensors

16 tensors, 2,381,264 entries changed (2.1749% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| bert.encoder.layer.11.attention.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.output.LayerNorm.weight | 8 | 0.0104 | 2.828 | 27.713 |
| bert.encoder.layer.11.attention.output.dense.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.output.dense.weight | 768 | 0.0013 | 82.898 | 55.426 |
| bert.encoder.layer.11.attention.self.query.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.query.weight | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.value.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.attention.self.value.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.intermediate.dense.bias | 3,072 | 1.0000 | 6.556 | 5.159 |
| bert.encoder.layer.11.intermediate.dense.weight | 2,359,296 | 1.0000 | 60.267 | 60.183 |
| bert.encoder.layer.11.output.LayerNorm.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.output.LayerNorm.weight | 0 | 0.0000 | 0.000 | 27.713 |
| bert.encoder.layer.11.output.dense.bias | 0 | 0.0000 | 0.000 | 0.000 |
| bert.encoder.layer.11.output.dense.weight | 11,968 | 0.0051 | 81.453 | 0.000 |
| bert.pooler.dense.bias | 8 | 0.0104 | 0.336 | 0.336 |
| bert.pooler.dense.weight | 6,144 | 0.0104 | 13.493 | 13.231 |

## Editor's final answer

I submitted a 10-layer model with 95,312,648 parameters, 12.9% smaller than the original 109,488,392. The best accuracy estimate is about 98.5–98.6% (98.53% in 20-fold cross-validation and 98.13% in 5-fold across 4 seeds). The original scores 98.31% on test, with 97.8% in 5-fold and about 98.4% leave-one-out. I couldn't safely shrink it further; I tested and rejected feed-forward pruning.

**Layers kept:** original layers 0–8 unchanged, plus layer 11 rebuilt into a pooling-and-classifier layer. Layers 9 and 10 are dropped and all 3,072 feed-forward units are kept.

**Why this depth.** I fitted a kernel ridge classifier on the token-averaged hidden states after each layer of the original model. Up to layer 7 they score about 97.3–97.9% (leave-one-out on the 1496 practice sentences). After layer 8 that jumps to 98.66%, and later layers add nothing. The 8 "reserved" dimensions were left out of every evaluation because they hold the previous editor's ridge scores, which were fitted on these same practice sentences. So layers 0–8 plus one layer to pool and classify is the shallowest design with enough margin.

**What the rebuilt layer 11 does:**
1. **Pooling:** every attention head's query is zeroed, so [CLS] receives the plain average of the layer-8 token states. Value is the identity and the output is 5× the identity. [CLS] then becomes LayerNorm([CLS] + 5 × average), with the reserved dimensions zeroed.
2. **Classifier in the feed-forward block:** units 0–1495 each correspond to one practice sentence. Unit i computes GELU(z(x)·z(c_i)/760), where z is a per-dimension z-score folded into the weights and bias, and c_i is that sentence's pooled vector.
   - The output weights hold the coefficients of a closed-form kernel ridge solve (penalty 0.01), written into the 8 reserved dimensions as class scores. Unused units are zeroed.
   - Because the solve uses this exact GELU kernel, the cross-validated model is exactly the model that ships.
3. **Pooler and classifier:** pooler rows 0–7 read the 8 score dimensions with their shared mean subtracted (which cancels the LayerNorm offset), followed by tanh. The classifier stays the identity on pooled dimensions 0–7, so the prediction is the argmax of the ridge scores.

**Checks:** I confirmed that the pooling layer matches its closed-form formula to about 1e-7 relative error on 1024 sentences. That let me compute the classifier's inputs for all 1496 sentences from earlier reads of the original model. The harness's logits match my numpy expectation to 1e-5 on 1034 sentences, with every prediction agreeing.

**What I tried and rejected:**
- **Feed-forward pruning:** I ranked units with an approximate importance estimate and compensated the output bias for the removed units. Keeping 2,560 or 2,816 units per layer made the features after layer 8 measurably worse on the same 1024 sentences. Kernel ridge leave-one-out fell by about 0.5–0.8 points, and 5-fold by about 0.3–0.8 depending on the head. The importance scores are fairly flat across units, so there is nothing cheap to cut. I reverted all pruning edits.
- **Dropping more layers:** dropping layers 5 and 7 cost about 0.8 points, and shallow designs top out around 97.9%.

The original model stays unchanged except layer 11 and the pooler. No training was used: every fit is a closed-form ridge or kernel ridge solve on activations of practice sentences returned by forward passes.

## Size and latency

| total params | non-embedding | layers | FFN width | checkpoint | reduction vs 109,488,392 |
|---|---|---|---|---|---|
| 95,312,648 | 71,475,464 | 10 | 3072 | 381.3 MB | 13.0% |

CPU latency (4 threads, arm, median of 7): batch 1: 29.819 ms/sentence, batch 32: 12.288 ms/sentence

Accuracy target 97.8%: **FAIL** (test accuracy 97.41%)

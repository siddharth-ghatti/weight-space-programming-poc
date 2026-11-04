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
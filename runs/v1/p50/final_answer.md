I turned BERT into the insurance classifier using only closed-form math: ridge regressions fit on hidden states of practice sentences, then written directly into the weights. Nothing was trained iteratively.

**Expected accuracy:** about 97.8% on new sentences, from 5-fold out-of-fold cross-validation (5-fold CV) on the practice set. The final model gets 100% on all 1496 practice sentences, but the head was fit on those same sentences, so that number only shows the edits work as designed. It says nothing about the test set.

**What I found first:** I read hidden states from every layer for all practice sentences. The average over all tokens at any layer separates the 8 labels much better than the [CLS] vector does: about 96% with a ridge classifier. Combining an early, a middle and a late layer reached about 97.8–97.9%.

**How the edit works:**
1. **A reserved channel.** I set aside 8 of the 768 hidden dimensions to hold class evidence. At the end of layer 1, I zero their LayerNorm scale and shift so they start empty. In layers 2–10 I cut every read and write of those 8 dimensions, so the rest of the network behaves as before and nothing disturbs the channel.
2. **Carry heads in layers 2, 6 and 10.** In each of these layers I repurposed one attention head. Its query is zero, so it attends equally to every token and effectively computes the average token vector. Its value and output weights turn that average into 8 ridge scores (one per label), added into the reserved dimensions. A small gain keeps those dimensions from upsetting LayerNorm.
3. **Layer 11 as a pooling layer.** I rewrote its attention so [CLS] receives roughly 2 × (average of layer-11 token states) + its own state, with the feed-forward output turned off. The final [CLS] vector therefore holds the late-layer average plus the 8 carried scores.
4. **Pooler and classifier.** Pooler rows 0–7 equal 0.5 × [ridge weights on the final [CLS] vector + a decode of the reserved channel]. The decode subtracts the channel's mean across the 8 dimensions, because LayerNorm adds a shared offset to all of them. The classifier simply selects pooled dimensions 0–7. Since tanh is monotonic, the predicted label is the argmax of the ridge score, and the tanh outputs stay well short of saturating (they range from -0.11 to 0.55).

**How I fitted the weights:** First I installed the carry heads and the aggregation layer. Then I read the edited model's real activations and fitted one joint ridge (penalty 300, selected by cross-validation) on the three carried layer averages and the final [CLS] vector. The fit used each sentence's measured carry scale, so it matches how the network actually behaves.

**Checks:** Running all 1496 practice sentences through the edited model gave outputs equal to my numpy prediction (tanh of 0.5 × ridge score) to within about 1e-4 (2e-4 on the first read, 8e-5 on the second). Every prediction was correct, and the smallest top-1 margin was 0.175.

**Bug found and fixed:** An earlier version ignored the shared offset in the reserved dimensions. That saturated tanh and dropped accuracy to 51%; the mean-subtracting decode in step 4 fixed it.
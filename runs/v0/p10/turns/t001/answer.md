## reasoning summary

I'm planning to fit a closed-form linear classifier, like ridge regression or LDA, on hidden states pulled from forward passes over all 1496 practice sentences. I can read the states in one or two passes, since each pass takes up to 1024 sentences. I'm still working out how the pooled representation is built from the layer weights.

I'm weighing whether to alter the pooler so the classifier gets mean-pooled information, but that means editing the encoder and seems messy. The simpler route is to keep the CLS state from layer 12, and since tanh is close to linear for small weights, the pooler can act as a linear projection, with ridge or LDA on top.

I'm considering making every head in the last layer attend uniformly by zeroing the query weights, so CLS ends up with a mean of the value vectors. That still leaves CLS's own residual in the mix, so I'd need another way to get a truly mean-pooled feature, perhaps by adjusting the value and output projections so the attention output dominates.

I'm settling on reading features from the edited encoder itself, so whatever the modified last layer computes gets captured exactly. The pooled output is linear in the classifier, so fitting on those features is exact rather than approximate. My next step is designing the encoder edits to get good sentence features at the last layer's CLS position, then setting the pooler.

I'm going to fit the classifier on the edited encoder's own pooled output, so the features are exact. Before that, I'll try a base read of per-layer CLS and mean-pooled hidden states across all 1496 practice sentences, in two passes, and run cross-validated LDA and ridge in the sandbox to see which layer and pooling works best.

I'm weighing whether mean-pooled hidden states from around layers 8 to 12 would work best for the classifier, since I can only feed the 768-dim CLS output of the last layer through the pooler. I'm also considering a bag-of-words style approach on token embeddings, since the task looks keyword-driven, though contextual mean-pooled layers probably do better. Next I need to figure out how to get mean pooling at the last layer's CLS position.

Setting the query weights to zero makes attention uniform, so the CLS token's attention output becomes the mean of all token values; I'm now working out how to set the value projection to identity so each head passes the average of the token states through.

I'm setting the output of the attention block to the scaled mean of the token states, so the residual from the CLS token barely changes it. Once that's in place, I'll reset the layer norm weights and then move on to the feed-forward block.

I'm working out that zeroing the query weights gives uniform attention, and that the last layer's feed-forward block can be turned off so the normalized mean comes straight through as the CLS feature. I'm also thinking about whether an earlier layer's mean would work better, which would mean making the layers after it pass through unchanged.

I'm checking whether the layer-norm step still holds once the residual input is already normalized, since renormalizing it won't give back the same values, so an exact pass-through isn't straightforward.

I'm checking whether a per-token LayerNorm distorts the mean if I only zero the attention output, since LayerNorm rescales each token differently. Zeroing the attention output and the FFN output in later layers would let CLS pass through, but the normalization still isn't a true identity.

I'm checking whether the final LayerNorm can be set to weight 1 and bias 0, with the later layers' attention and FFN outputs zeroed, so the CLS state passes through unchanged. I'm also weighing whether mean pooling from a middle layer would beat the last one, though I can only carry 768 dimensions through the pooler.

I'm weighing whether a mean-pooled feature from a middle layer would do better than the last one, but a true multi-layer sum isn't possible without messing up the CLS token. Keeping it simple seems wiser. I'm also thinking that for this keyword-heavy task a 768-dim mean-pooled input with LDA on 1496 examples will need regularization, so I'll cross-validate it.

I'm setting the pooler to a small scaled identity so that tanh stays nearly linear, then reading the CLS hidden state at layer 12 for the sandbox fit.

I'm planning to read the pooled output straight from the pass rather than recomputing tanh in the sandbox, so the features stay exact. I'll split the 1496 practice sentences into two passes, reading mean and CLS hidden states for every layer, then compare layers by cross-validated LDA and ridge.

Next I'm weighing whether the mean over tokens should exclude CLS and SEP, since my uniform attention includes them, which may cause a small difference. I could skip the two base read passes and go straight to the edited model, but knowing the best layer to pool from is worth the cost. After that I'd read pooled outputs from the edited model on all 1496 sentences, fit the classifier, and verify with one prediction pass.

I'm weighing whether the base read's mean over hidden states could stand in for the edited model's pooled output, since that would be a closed-form shortcut. I'd rather not, because computing layer norm and the linear pooler by hand on read activations is close to reimplementing the forward pass. Reading the edited model's pooled outputs directly is exact and safe. I'm also unsure whether the mean includes the CLS and SEP positions, which would change what the features actually capture.

I'm weighing whether a nonlinear classifier could beat the linear one on the pooled 768 features, since the pooler's tanh already gives one nonlinear layer. Random features with closed-form ridge would be an option, but a linear fit may be enough, so I'll check baseline accuracy first.

I'm considering whether mean pooling from a middle layer like 6 to 10 would beat the last layer, and whether concatenating several layers' means could help. The catch is that zeroing the query weights in one layer would also change how every non-CLS token attends, so stacking multiple layer means into CLS isn't straightforward.

I'm weighing whether I can make only the CLS position attend uniformly, since zeroing the query weights would change every token's attention, and I'd rather keep the other tokens intact. Repurposing one head would also disturb them somewhat. Summing layers directly doesn't look feasible, so I'll stick with a single-layer mean, pick the best layer by cross-validation on the base reads, and only consider combining two layers if the gain turns out to be large.

I'm weighing whether the CLS output could blend the CLS state with the mean of all tokens by scaling the attention output, which looks workable. Pooling a mix of two layers would need every token's processing to stay the same, and that's hard to keep while only the CLS position changes, so I'll stick with a single layer. I'm going to set up notes and check the practice data and tokenization lengths before running any passes.

I'm setting up the read so each layer's mean and CLS hidden states come out as 768-dim arrays, which works out to roughly 80MB for the first batch of 1024 sentences.



## message

Strategy: run the encoder for real (read passes) to get activations, then fit the classifier with closed-form math in the sandbox. Steps:
1. Read base hidden states (CLS and mean-pooled, all layers) for all 1,496 practice sentences. That takes 2 read passes because each pass holds at most 1,024.
2. Use cross-validated closed-form classifiers (ridge/LDA) to find which representation separates the 8 labels best.
3. Edit the last encoder layer(s) so the [CLS] output carries that representation, for example by making attention uniform so [CLS] becomes a mean-pool. Then read the edited model's actual `pooled` outputs and fit the classifier head in closed form on them.
4. Check the result with a predict pass.

First I'll look at the data and set up notes.

## tool_use: run_python

```json
{
 "code": "\nimport json, collections\nP = practice()\nprint(len(P), P[0])\nprint(collections.Counter(p['label'] for p in P))\ntoks = tokenize([p['text'] for p in P])\nL = [len(t['ids']) for t in toks]\nprint('len stats', np.min(L), np.mean(L), np.max(L))\nfor p in P[:12]: print(p['label'], '|', p['text'])\nprint(LABELS)\nopen('notes.md','w').write(\"\"\"# Notes\n## Strategy\n- Read base hidden states (cls + mean, all layers) for all 1496 practice -> CV closed-form probes per layer.\n- Edit last layers so CLS = mean-pool of best layer (uniform attention via zero Q), pass-through after.\n- Read edited pooled outputs, fit ridge/LDA classifier closed-form, verify with predict.\n\"\"\")\n"
}
```


## reasoning summary

I'm comparing the combined layer 0 and layer 11 mean result against each alone, and I want both signals inside the CLS path. I'm noticing the classifier only has to produce 8 logits, and since the mean over tokens is linear, the per-token contributions from both layers can be folded together before averaging.

I'm checking whether the layer 0 projection could be carried forward through each token's residual stream, but LayerNorm's per-token rescaling would break the linearity I'm relying on. I'm also weighing a variant where CLS collects the layer 0 term early, though that would change what the other tokens see in later layers. Another option is to dedicate a final-layer head to the combined projection, since the mean over tokens is linear.

I'm weighing whether the embedding-layer projection can survive through the residual stream to the final layer, but per-token LayerNorm rescaling and the attention and FFN writes at every layer would scramble that direction. I'm considering whether a k-dimensional bag-of-embeddings signal could be kept in a subspace the network barely uses, though that seems too complicated to rely on.

I'm now weighing whether to repurpose the final encoder layer or an earlier one, since the CLS token's attention there could carry the layer 0 signal forward, but that would also reshape what the other tokens see.

I'm weighing whether the layer 0 signal could survive to the end: making one head in the early layers attend uniformly so every token carries the mean of the embedding projection, though that would perturb everything downstream unless the added vector stays small. I'm also questioning whether the (0,11) gain, 95.6 to 97.4 in cross-validation on 1024 examples, is real and would hold on test data.

I'm checking whether a bag-of-embeddings signal from layer 0 could reach the final CLS token, and I'm wondering if the (0,11) gain of 95.6 to 97.4 would hold up beyond cross-validation. Trying the (10,11) pairing next: it reached 96.4, a smaller gain.

Ridge on the layer 11 states alone reaches 95.6, while adding layer 0 reaches 97.4, so the layer 0 information doesn't come through linearly in the mean of the final-layer states. I'm now considering whether the layer 0 features could be computed per token inside the early FFNs, though that would overwrite each token's representation.

I'm considering a middle ground: carrying only a low-dimensional projection forward, since the (0,11) classifier needs just the 8-dim mean of the layer 0 projection. The difficulty is that per-token signals have to survive eleven LayerNorms and additive updates before they reach CLS, so I'm looking at whether CLS in the first encoder layer could collect that bag-of-embeddings signal and carry it through its own stream.

I'm looking at whether CLS in layer 0 could get a query that differs from the other tokens, so it attends uniformly to all tokens and gathers the mean embedding projection. The catch is that the query weights are shared, so I'd have to repurpose one of the existing heads in that layer, which would change what the other tokens see.

I'm wondering whether the CLS token in an early layer could attend uniformly while the other tokens keep their original attention, but the query weights are shared, so that's hard to do cleanly. I'm also considering whether BERT's residual stream has dimensions that go largely unused and could carry the signal, though I'm not sure that's a safe bet. Still weighing a way to raise the 95.6 baseline.

I'm weighing whether mean pooling should include CLS and SEP or only the content tokens, since I don't know what the harness does. I'm also thinking about the final layer itself: if its FFN output is zeroed and the attention output is scaled up, the CLS representation would be roughly a normalized mean of the value-projected token states.

I'm thinking about how to get the layer 0 bag-of-embeddings signal into the final CLS. With the last layer's attention uniform and its FFN zeroed, the CLS output becomes a normalized mean of value-projected layer 11 states, so the classifier stays close to the 95.6 baseline. The tanh pooler could also be used as a random-feature layer with ridge on top, which might add a little nonlinearity.

I'm asking whether the layer 11 uniform attention could draw on token states that already carry the embedding information, perhaps by changing encoder.layer.10. But the embedding signal would still have to pass through every layer's updates and normalizations, so I'm wondering how much token identity actually survives in the higher layers, and whether it could be recovered linearly.

I'm checking whether a skip connection can carry the layer 0 signal forward, but attention only sees the current layer, so that won't work. Another option is to repurpose one head in encoder layer 0 so CLS attends uniformly and gathers the mean embedding, though every token would then attend uniformly through that head too, changing what they see.

I'm weighing a version where the final layer's value projection carries the embedding signal, since the readout there can be any linear function of the layer 11 states. The final FFN, applied only at CLS, could add some nonlinearity before the pooler. Using two layers instead of one for the carrying step might be another option.

I'm considering whether the later layers could do the pooling instead: keeping the attention in layer 10 as it is so the layer 11 states stay contextual, and using layer 10's FFN to produce features worth averaging. The catch is that this FFN only sees the post-attention state at layer 10, and a ridge on layer 10 alone already gets about 96%. I'm also thinking about whether excluding CLS and SEP from the average, or weighting the tokens unevenly, could help, though I'd want that weighting to stay closed-form.

Checking whether the ridge baseline holds up with LN-normalized features or an LDA with shrinkage, and I'll test on the remaining 472 examples later. Another idea is to have CLS carry the mean across layers, with the later layers mostly leaving it alone, though changing CLS's values would also shift what the other tokens attend to.

I'm considering whether CLS in the first layer could attend uniformly by giving it a constant query, so it collects the mean of the token embeddings directly.

I'm reconsidering whether a uniform attention head in layer 0 could work: every token would get the same mean vector, so CLS can't be singled out without a nonlinearity. Still, that shared mean survives into the final average, though later layers would reshape it. Token identity at the top layer is largely recoverable, since the MLM head reconstructs input tokens from the final hidden states.

I'm weighing whether the final layer's value projection can carry the embedding signal, with the 95.6 baseline in mind. The plan is a basic pipeline first: uniform attention in the last layer, checked on the held-out data, then enhancements with the remaining passes.

Next I'm considering whether an enhanced readout could also use the lexical signal from layer 0, which means thinking about the value vectors in the final layer, since those are linear in the layer 11 states.

I'm checking whether the mean of h0 can be predicted linearly from the mean of h11, though a ridge on mean h11 already covers every linear function of that mean, so that test alone gives nothing new. The real option is a different family of features, such as the mean of a nonlinear function applied per token at layer 11, or features from an earlier layer.

I'm considering whether a nonlinear per-token function could be added to each token's residual stream at layer 10, since token identity might survive in the residual stream even though the mean-based ridge is already linear in that mean. Without token-level states for all items, I'd have to pick random features blind, and I'm wondering whether testing the accuracy of layer 0 alone at different regularization strengths would be a cleaner next check.

I'm comparing the multi-layer combinations: (0,11) reaches 97.4, and (0,6,11) gets 97.6, so mixing layers gives about a 1.5-point gain. I'm now wondering whether CLS could accumulate means from several layers as it passes through them, though the intermediate layers might scramble what it has gathered. Using a different token as the accumulator doesn't work, since only CLS is read out at the end.

I'm considering a multi-layer variant where CLS accumulates means at layers 10 and 11, though uniform attention at layer 10 would also change the layer 11 states for every token.

I'm weighing whether the last layer's per-head value projections, which are linear in each token's layer 11 state, could carry more than a single linear readout. Rather than adding another repurposed head, I'd try different feature families on the mean: normalized per-token features, ridge under alternative normalizations, LDA with shrinkage, and a random tanh layer on top of the pooled mean.

I'm considering a content-based attention head in the final layer: the query is a constant bias, so each token gets a weight from its key, letting the pooling emphasize tokens relevant to each class. Several heads could each target a different class direction, which would make the pooled features nonlinear and possibly help with key-phrase detection.

I'm weighing a content-based attention head for the final layer, where the key direction comes from a per-token class score built from the mean-level ridge weights. The softmax over those scores would then emphasize class-relevant tokens, and I'm checking whether that's feasible without per-token states for all items.

I'm going to make the final layer's attention uniform for CLS, then I'll check that base pipeline before adding any nonlinear pooling. Since the eventual read of the edited model gives exact final-layer CLS features, the classifier fit stays closed-form.

I'm setting up the final layer so every head attends uniformly over non-padded tokens, with the value projection as identity so the concatenated output is the mean of the layer 11 states. I'm also considering the output projection as a scaled identity, or just keeping the CLS state added in, so the features mix the CLS vector with that mean.

I'm considering keeping the output projection as a large scaled identity so the CLS residual is suppressed, which leaves only the mean-based features. I could also compress the mean to a few hundred PCA dimensions and check ridge CV on that, though the residual connection means the CLS part can't be shrunk the same way.

I'm setting the final layer's feed-forward to zero and its second layer norm to identity, so the output is just the layer-normalized mean. Then I'll let the pooler and classifier sit on top of that, with the CLS residual damped by the large output scale so it contributes only about one percent.

I'm checking whether the layer 11 mean could be normalized in closed form, since LayerNorm would strip out the scale effects from those outlier dimensions. I'm also weighing whether to keep a bit of the CLS state through the residual, but it would add noise that the mean-based features don't need, so I'll leave it out.

I'm weighing whether to split the 12 heads between a full uniform mean and some content-weighted pooling. Uniform heads with identity values need all 768 dimensions to carry the full mean, so giving some heads to attention pooling would mean compressing the mean to around 512 dimensions, for example with 8 heads of 64 each.

I'm weighing how to set the key direction for the content-based heads: each head would take a class score per token, taken from the mean-level ridge weights, so the softmax over those scores picks out class-relevant tokens. I'm also planning to split the heads, using some for the uniform mean and others for this attention pooling, and checking whether ridge CV holds up on a PCA-compressed mean.

I'm thinking about checking whether per-token class scores could drive attention pooling, but I'd need token-level states for many sentences to evaluate that offline, and a single pass only covers 64 texts. So I'm considering testing it directly by building the edited model with both the uniform mean and the class-score heads.

I'm checking the attention-based pooling against the uniform mean: both sets of heads go into one edited read, and ridge CV on the full features gets compared with CV on only the mean-based dimensions. The catch is that the attention directions come from the same 1024 items, so the CV could look better than it really is. That leak is mild, since the directions only set the attention weights.

I'm weighing whether the bag-of-embeddings idea holds up, since the (0,11) combination gave the biggest gain, and I want to know how much lexical identity survives in the layer 11 states.

I'm wondering whether the attention pattern itself could act as a lexical detector, since keys are linear in the layer 11 states and those carry token identity. I'll first run some basic checks offline: ridge CV on PCA-compressed mean features at a few sizes, whether normalizing the mean changes anything, and whether the harness mean includes the CLS and SEP tokens.

I'm going to skip any embedding-layer computation, since reimplementing that part of the forward pass isn't allowed and I'll just read the edited model's outputs instead. The layer index works out cleanly: hidden state 11 is the output of encoder layer 10, which is the input to layer 11, and its mean is still very similar in quality to the layer-10 mean.

I'm considering whether to also bring in lexical information from layer 0, but the last layer's content-based heads can't see it, so I'll set that aside for now. Another idea is to read out at the very first encoder layer and let a uniform head there inject class-direction scores into the residual stream, which would then pass through all eleven layers.

I'm dropping the lexical idea for now, since the mean at layer 11 may not keep enough token identity to matter. Next I'll run the quick offline checks: ridge CV on PCA-compressed mean features at a few sizes, and whether normalizing the mean changes anything.



Concatenating layers gains about 1.5 points, but wiring layer-0 into the final attention is hard, so my baseline uses only layer 11. Before building it, I'll check offline that LayerNorm and PCA compression keep accuracy.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]; folds = make_folds(1024)\nm = load('mean',11)\ndef LN(X): \n    X = X - X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\nlams=(0.05,0.1,0.2,0.4)\nprint('m', ridge_cv(m,y,lams,folds))\nprint('LN(m)', ridge_cv(LN(m),y,lams,folds))\n# PCA compression\nmc = m - m.mean(0); U,S,Vt = np.linalg.svd(mc, full_matrices=False)\nprint('var explained', [(k, round((S[:k]**2).sum()/(S**2).sum(),4)) for k in [64,128,256,384,512]])\nfor k in [128,256,384,512]:\n    Z = mc@Vt[:k].T\n    print('pca',k, ridge_cv(Z,y,lams,folds))\n# no-standardize ridge (raw)\n"
}
```


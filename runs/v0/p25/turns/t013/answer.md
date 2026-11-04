## reasoning summary

Gains about half a percent over the baseline, so that's a small but real improvement. Now I'm wondering whether the lower-layer mean could also go into the final representation, and I'm considering layer 11's value projections as a way to bring in more lexical information.

I'm checking whether mean0 can be linearly predicted from mean11, since if it is, the extra information might already be there and LDA is just not using it well. Linear combinations of token values stay linear after uniform pooling, so I'd need nonlinear per-token features, perhaps from the layer 10 feed-forward network, before pooling.

I'm weighing whether layer 10's feed-forward network could be repurposed to add nonlinear per-token features before pooling, since layer 11 is only used for pooling now. The catch is that the residual stream is already fully used, so new features would mix with existing ones, and it's unclear which per-token features would actually help without token-level data.

I'm wondering whether CLS at layer 11 could carry the mean0 information, since its state is directly in the residual stream and every other token's layer 1-10 states attend to it. That depends on how much the other tokens actually attend to CLS in BERT, which I haven't checked yet.

Making CLS's keys unattractive to other tokens would need CLS-specific key features, since a key bias shifts every key equally and cancels in the softmax. I'm wondering whether CLS can be isolated this way while its own accumulated information stays intact.

I'm looking at whether CLS could be isolated so other tokens' attention to it is suppressed, which would mean putting a consistent query component into one dimension and a large negative value into CLS's key in that dimension. That would cost some of the 64 dimensions per head, so it changes the head's capacity.

I'm considering whether a more elaborate attention modification to isolate CLS could work, but it would mean repurposing dimensions across many heads and layers, which seems too intricate. I'm also wondering whether BERT's attention to CLS is functionally important, since it may act as a no-op sink. Other pooling options at layer 11, like excluding CLS and SEP or using multiple attention patterns, might be more worth exploring.

I'm considering whether attention patterns that focus on the first few tokens could recover word-order information that uniform pooling discards, since opening phrases like definitions or exclusions are strong signals.

I'm planning to test the designed-M version first, combining the CLS state at layer 11 with a linear mapping of mean11, and reading pooled features for all 1496 texts across two passes. Each candidate idea costs about two passes, so I can afford only five or six tests within my budget of 23.

Now I'm weighing whether to use a designed mapping M with direct weights or simply refit the classifier. The concern is leakage: since M is fit on the same 1496 texts, cross-validating on the pooled features would look optimistic, so the direct approach avoids needing a refit at all.

I'm working out the layer-11 attention edit so the CLS output carries the mean of the token states. I'm setting the value and output maps so the pre-LayerNorm vector is the CLS state plus a mapped mean, plus a constant offset that stabilises the normalisation.

I'm working out how the layer-11 attention output can be set so the CLS vector absorbs a scaled copy of the mean states, with a constant offset to keep the LayerNorm well-behaved. For large offsets the normalisation becomes nearly linear, but the signal part gets small relative to the constant term, so that trade-off needs checking. I'm also noting that if the feed-forward branch is zeroed out, the final output is just the normalised vector itself.

I'm working out how the pooled output could be made linear: the layer-11 CLS vector would carry a mapped mean of the token states, with a constant offset to keep normalisation stable, and a final projection that strips the constant direction and centres the signal. Then I'm wondering whether all that linearity is really needed, or whether I should just read the actual pooled features and fit a classifier on them directly.

I'm weighing a designed mapping M against a plain classifier refit, and the label-dependent M would make cross-validation optimistic, though the final classifier would still be fine. Without label leakage, a simple identity mix only gained about 0.4% (0.970 versus 0.975), which might be worth it. I'm now wondering whether layer 11 could compute the final class logits directly, since the classifier only needs eight numbers.

I'm considering whether layer 11's pooler could read the CLS state and implement a linear mapping directly, since only eight classifier outputs matter. Extra views beyond what's already in the residual stream would need accumulation across layers, but the last layer's input only carries layer-10 information, so layer 10 can't add new views on its own.

I'm checking whether mean11 linearly encodes the mean0 signal, since the combination of the two views does better than either alone. Feeding mean0 into every token through the layers doesn't look clean, though.

I'm considering keeping some residual dimensions reserved for a layer-0 projection of each token, and zeroing out every layer's writes to those dimensions so the signal isn't drowned by updates. The open question is whether layer normalization and the accumulated updates would still leave that signal recoverable by layer 11.

I'm working through how to keep the layer-0 signal alive: reserving a few residual dimensions and zeroing out every layer's writes and reads to them. The per-layer LayerNorm rescaling by each token's standard deviation complicates things, and dropping those dimensions would also alter BERT's own computation, so I'd need dimensions that matter little.

I'm wondering whether layer 0 could pool the bag-of-embeddings signal directly for the CLS token and just accept the side effects on the other tokens, and I'm planning to test that empirically.

I'm wondering whether the CLS token could carry mean-zero signal through layers 1 to 10 when it's the only token that gets the layer-0 pooling, since the residual stream is renormalised at every layer. The open question is whether that layer-0 information would still be recoverable at layer 11 once the updates have been added on top.

Now I'm checking whether the CLS token's layer-0 attention can be made uniform, so it pools the mean of the token embeddings without disturbing the other tokens.

I'm considering a layer-0 trick: sacrifice one attention head so it attends uniformly for every token, which would add the same mean-pooled vector to all tokens as a sentence-level shift. That avoids disturbing the CLS-only pooling, though it changes every token's representation, so I'd need to check its effect empirically.

I'm weighing whether a per-token route could work instead, where each token's layer-11 state keeps a clean bag-of-embeddings signal. Reserving residual dimensions for that is complicated, since layer norms and the other layers' updates would disturb them, so I'm looking at whether a cheaper way to recover the mean-0 information at layer 11 exists.

I've settled on a first version: freeze layer 11's query so attention is uniform, set the value and output projections so the CLS vector carries the mapped mean, and leave the output bias at zero. I'm still unsure whether to use the designed mapping or a simple scaled identity, since the designed one makes the cross-validation optimistic.

I'm settling on a simpler form: the final CLS state is the layer-11 vector with the mean-mapped term added, normalised with no scale or shift and no feed-forward. The pooler would be a small scaled identity with zero bias, so I'd read the pooled features and fit a classifier on them. Since the normalisation roughly preserves the linear mix, the fit should land close to the direct logits, though I still want to compare the two on held-out data.

I'm weighing a designed mapping that uses all the data against a symmetric, label-free alternative, since the designed version only leaks into cross-validation through a low-rank term. A refit head on the pooled features with mild regularization could work, and I'd like to compare that against direct logits on held-out data.

I'm settling on version one: the layer-11 pooling uses the designed mapping fit on all the data, followed by a scaled-identity pooler and a ridge refit on the pooled features. I'll check that training accuracy comes out high and then confirm the harness predictions match. I'm also wondering whether keeping layer 11's feed-forward block and the original layer norms would add useful nonlinearity to the CLS state.

I'm checking whether the mean includes the CLS and SEP tokens, since that decides whether uniform attention reproduces mean11 exactly. Then I'll set the value projection to the identity and the output projection to the designed mapping, with zero bias.

I'm setting the layer-11 attention to uniform by zeroing its query, with the value projection as identity and the output projection taking the mean mapping. The feed-forward output gets zeroed, and both layer norms are reset to identity scale and zero shift. The pooler will be a scaled identity with zero bias, so the final CLS state is just the layer-normalised sum.

I'm checking the per-dimension mean and standard deviation of the layer-11 CLS and mean vectors, so I can see how large the outlier dimensions are. Then I'll pick a scale that keeps the pooler inputs mostly under about 1. I'm also considering a standardizing pooler, with per-dimension scaling and bias estimated from the statistics of the pre-normalized vector.

I'm going to look at the per-dimension statistics of the layer-11 CLS and mean vectors, mainly to find the outlier dimensions, and then pick a pooler scale that keeps most inputs near magnitude one.



The designed mixing reaches about 97.5%, versus 97.0% for plain mean11, and it can be realized exactly in the last encoder layer. I'm checking the scale of x first so the pooler stays in tanh's near-linear range.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nWt,b=ridge_fit(np.hstack([c11,m11]),y,100)\nA=Wt[:,:768]; B=Wt[:,768:]; c=2.0\nM=c*np.eye(768)+np.linalg.pinv(A)@(B-c*A)\nX=c11+m11@M.T\nprint('train acc direct', (np.argmax(X@A.T+b,1)==y).mean())\nPc = X - X.mean(1,keepdims=True); Ln = Pc/ Pc.std(1,keepdims=True)\nprint('LN(x) abs max per dim top:', np.sort(np.abs(Ln).max(0))[-10:])\nprint('LN(x) per-dim mean abs top', np.sort(np.abs(Ln.mean(0)))[-10:], 'typical std', np.median(Ln.std(0)))\nprint('M norm stuff', np.linalg.norm(M-c*np.eye(768),2), np.abs(M).max())\nprint('x norm', np.linalg.norm(X,axis=1).mean(), 'std of LN scale', (Pc.std(1)).std()/Pc.std(1).mean())\nnp.save('M_v1.npy',M.astype(np.float32))"
}
```


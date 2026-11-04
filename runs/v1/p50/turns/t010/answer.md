## reasoning summary

Combining layers seems to help: the mix of layer 0 and layer 11 gets about 97.5%, and adding layer 6 or averaging everything reaches roughly 98%, which leaves about 2% of room over a single layer. That makes a multi-layer mean worth trying to build. The open question is how to get the mean of the embedding output and the mean of a late layer into the final CLS representation.

I'm weighing two routes for getting the layer means into the CLS output. The first is to fit a ridge classifier on the concatenated layer means, so each layer contributes a linear projection that can be folded into the CLS path. The difficulty is that BERT's residual stream has no spare dimensions and LayerNorm reshapes whatever gets written there.

I'm weighing whether a spare attention head could write the projected layer means into a reserved part of the CLS residual stream, though BERT has no unused dimensions and LayerNorm rescales whatever gets added. Routing the token-level projections through the final layer instead runs into the same obstacle, and the m11 probe result suggests the layer-0 signal isn't fully recoverable from the last layer alone.

I'm weighing whether a spare attention head could carry the projected layer means into the CLS path, but BERT has no unused residual dimensions and LayerNorm rescales anything written in. A shallower design or a bag-of-embeddings plus one contextual layer might capture most of the gain, since m0+m6 already sits close to the all-layer mean.

I'm considering a cleaner route: a dedicated attention head could make the CLS token attend uniformly to the other tokens at an early layer, so its residual state picks up a mean of token features. The open problem is that CLS's query is computed from its own state, so getting only CLS to aggregate while other tokens process normally needs a way to make its query special.

I'm working on getting CLS to attend uniformly at layer 0 for one head: its query is a constant, so I'd need to adjust the query bias so CLS's scores come out flat across keys while other tokens keep their queries mostly intact.

I'm checking whether a spare attention head could make the CLS token attend uniformly to the other tokens at layer 0, by zeroing out its query so the scores come out flat. The catch is that every other token's query gets perturbed a bit too, since the layer-0 embeddings share a common component.

I'm weighing whether the [SEP] token could serve as the aggregator instead, but it faces the same problem: BERT's heads already use [CLS] and [SEP] as attention sinks, so changing their value vectors would ripple into every other token. Maybe the cleaner route is to carry each token's own early-layer projection forward through the residual stream, then aggregate at the final layer.

I'm considering a layer bypass: run the network only up to some middle layer, make the later layers pass through unchanged except the last one, and have that final layer aggregate. The open question is whether layer-0 information can survive that route better than it does through the raw residual stream.

I'm considering whether a subspace of the residual stream, maybe 16 to 32 low-variance dimensions the pretrained model barely uses, could carry a separate early-layer signal, with the layer-0 projection written into it.

I'm checking whether the FFN can carry a per-token early-layer projection exactly, since a pair of GELU units gives a linear identity. The open issue is that the following LayerNorm rescales the whole vector, so a write into a reserved subspace would shift the mean and scale of everything else.

I'm checking whether the per-token 1/σ scaling in LayerNorm would undo a reserved-subspace write, since that scale differs for each token and layer. Keeping the subspace orthogonal to the all-ones vector would protect the mean, but the late-layer features could still take a hit from the extra interference.

I'm checking how the per-token LayerNorm scaling would affect a signal carried in a reserved subspace. Each token gets multiplied by its own factor at every LayerNorm, so the carried signal becomes a weighted mean rather than a plain one. That's still a usable bag-of-words feature, so the classifier could be fit on it.

I'm thinking about zeroing the columns of the downstream weights (query, key, value, and intermediate) that read the reserved coordinates, so the model ignores them. The catch is that LayerNorm's mean and variance would still pick up whatever is stored there, and those coordinates may have carried useful information originally.

I'm noting that subspace S would need its downstream read weights zeroed and its writes suppressed, and that doing so would discard whatever the original coordinates carried. I'm considering whether the LayerNorm gain γ, which is per-coordinate, would still mix the subspace with the rest unless I can keep γ constant across it.

I'm checking how LayerNorm's mean subtraction would react if the reserved coordinates held a signal, and the fix I'm considering is to store each value as a +v and -v pair across two axis-aligned coordinates so the mean stays put. The variance still picks up a small contribution from those coordinates, which should barely shift σ if the signal is small enough.

Now I'm weighing a simpler option: pick a different feature from the final layer, such as a different pooling or a combination with CLS features, instead of routing layer-0 information forward. Next I'm checking whether random nonlinear features of the final-layer mean could reach a similar accuracy.

I'm considering a fourth route: have CLS attend uniformly at some chosen layer so its state absorbs an average, and let the later layers read that CLS as a sink. That would change CLS's own state a lot, though, so I'd need to check what the other tokens pick up from it.

I'm considering whether CLS's own state could carry the early mean through the middle layers, since its final state would be a nonlinear mix of that mean and the context. I'd then combine the CLS residual with a linear projection of the mean in the last layer, so both feed the classifier rather than scaling the mean to dominate.

I'm trying to make CLS's query zero at layer 0 so it attends uniformly, but that also nudges every other token's query through their shared component. I'm also wondering whether CLS's input has a unique coordinate I could use to isolate the change.

I'm trying to make CLS's query vanish at layer 0 by finding a direction u that picks out the CLS coordinate while staying nearly orthogonal to every other token's embedding. Minimizing the squared projections onto the other tokens should give u proportional to the inverse covariance times e_cls, so the other queries barely move.

I'm now trying to make CLS's first-layer attention uniform across all heads, which would make its output a function of the mean of the layer-0 token states passed through the residual, LayerNorm, and FFN. I'd then be running that odd CLS state through the later layers, where other tokens attend to it.

I'm sticking with the reserved-subspace idea, since it's the more controlled option. The aim is for the final CLS features to include a small projection of the layer-0 mean alongside the layer-11 mean, with the final layer's uniform attention doing the aggregation.

I'm now looking at whether a few residual coordinates can be freed up, since the stream is fixed at 768 dimensions and nothing else persists across layers. The plan is to zero out k of them and measure how much the probe accuracy drops, and I'm noting that BERT's outlier dimensions may make some of these choices costly.

I'm weighing whether the early-layer signal could come through a shallower combination, so I'm checking whether pairing layer 0 with an early layer like 2 or 3 gets close to the 97.5% from layer 11.

I'm weighing whether a shallower stack could work instead, since the carry-forward problem only gets smaller with fewer layers. Another route is making the final layer aggregate from identical token states, but that still leaves only one layer's information available. What the layer-0 signal really contributes is lexical identity, which later contextualized states partly lose, so I'm wondering whether that could be approximated from the final token states another way.

I'm weighing whether the carry-forward problem is worth solving at all, since each layer's information is partly lost and the reserved-coordinate approach has a cost I haven't measured. The next step is to zero out k coordinates from layer 1 onward and run forward passes to see how much probe accuracy drops.

I'm considering whether the reserved coordinates could be avoided altogether, perhaps by using attention-sink tokens. I'm also wondering if the LayerNorm weights could be edited so that a signal fits along coordinates where γ is roughly constant, which would let a subspace carry the layer-0 information.

I'm considering whether the residual stream could be made axis-independent by choosing an arbitrary subspace instead of individual coordinates. That would mean projecting out the subspace in every reading and writing weight, but LayerNorm's diagonal γ would still mix coordinates unless it is constant on the subspace's support.

I'm now checking whether LayerNorm's per-coordinate scale γ blocks the idea, since the residual path carries γ⊙z + β forward and can't be absorbed into downstream weights. That forces an axis-aligned choice of coordinates, or a subspace on coordinates where γ is roughly constant. I'm also ruling out the all-ones direction, because LayerNorm's mean subtraction would wipe out anything stored there.

I'm considering limiting the carry to layers 0 through 5 so that only six layers need reserved coordinates, with layers 6 through 10 passing the state through unchanged, and I'm planning to compare the m0+m6, m0+m4, m0+m8, and m0+m11 probe accuracies against m0+m12. I also want to check how many dimensions the layer-0 mean actually needs.

I'm weighing a cheaper check: fit a ridge model on the layer-0 mean alone to get a few logit-style features, then add those to the layer-11 mean. The catch is that fitting that projection on training data would overfit, so I'll need to compute it within each cross-validation fold.

I'm going to test PCA directions of the layer-0 mean, at k = 8, 16, 32, and 64, concatenated with the layer-11 mean. I'm also wondering whether the full 768 dimensions of each just help through extra features and regularization, which would mean the carry isn't actually the source of any gain.

I'm checking how much signal gets lost as it passes through LayerNorm at each layer, since the per-coordinate scale γ divides by the token's standard deviation, which grows as sublayer outputs are added. I'm also considering how the cumulative shrinkage over the many LayerNorms would weaken a carried signal stored in reserved coordinates.

I'm considering whether the signal can survive all the LayerNorms in the stack, since each one rescales the reserved coordinates by a token-dependent factor that compounds over roughly 22 normalizations. I could pin the reserved gain to compensate, but the variation across tokens would still make the carried feature noisier than the clean layer-0 mean.

I'm considering keeping the signal's scale stable by storing it alongside a constant reference coordinate, so that each LayerNorm rescales both the same way and their ratio recovers the original value.

I'm settling on the idea of keeping the signal in one place and running it through parallel weights, so it passes the LayerNorms with less scaling variation.

I'm going to test a simpler variant: if m0+m1 does as well as m0+m11, the carry only needs to span a single layer, with layers 1 through 10 passing the state through and layer 11 aggregating.

I'm testing whether the layer-0 information can ride along through the stack by checking the m0+m1 pair, since the residual at layer 1 already carries h0 strongly, and if that works, I may not need reserved coordinates at all.

I'm going to compute the full matrix of layer pairs, starting with pairs that include layer 0 and pairs that end at layer 11, plus a few others, and compare them against the PCA-carried versions.



Combining layers reaches 97.5–98.0% versus about 96% for one layer, so it's worth engineering for. Next I'll map which layer pairs help most and how many PCA dimensions of the early signal need carrying.



## tool_use: run_python

```json
{
 "code": "lab = np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nM=np.load(WORK/'base_mean.npy')\ndef zs(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef ridge_cv(X, lams):\n    res={}\n    for lam in lams:\n        pred=np.zeros(N,int)\n        for f in range(5):\n            tr=folds!=f; te=~tr\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n            Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n            K=Xt@Xt.T; a=np.linalg.solve(K+lam*np.eye(len(K)), Y[tr]-Y[tr].mean(0)); S=Xe@(Xt.T@a)\n            pred[te]=np.argmax(S,1)\n        res[lam]=round((pred==lab).mean(),4)\n    return res\nfor b in [1,2,3,4,6,8,10,11,12]:\n    print('m0+m%d'%b, ridge_cv(np.concatenate([M[0],M[b]],1),[100,300]), flush=True)\nfor a in [1,2,4,8]:\n    print('m%d+m11'%a, ridge_cv(np.concatenate([M[a],M[11]],1),[100,300]), flush=True)\n# PCA-k of m0 appended to m11 (pca unsupervised on all data)\nZ0=zs(M[0]); U,S,Vt=np.linalg.svd(Z0,full_matrices=False)\nfor k in [8,16,32,64,128]:\n    X=np.concatenate([M[11], Z0@Vt[:k].T],1)\n    print('m11+pca%d(m0)'%k, ridge_cv(X,[100,300]), flush=True)\n"
}
```


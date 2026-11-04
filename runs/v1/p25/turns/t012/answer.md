## reasoning summary

I'm looking at which pairs work best: pairing the bag-of-embeddings layer with a late layer does better than the other combinations. Next I need to figure out how to carry that early signal through to the final CLS token, since the last layers are where I have the most room to change things.

I'm now checking whether M0 can be recovered from the layer-10 token states, since h0 isn't directly available there. Per-token nonlinear maps might work, but I'd need token-level data to fit one, and a linear map would only match the M10-only ridge unless the regularization differs enough to matter.

I'm wondering whether M0 is really unrecoverable from M10, or whether the gain is just a regularization artifact in the standardized space. A token-level linear map from h10 to h0 could be fit on far more samples, and then I'd concatenate M10 with that estimate. I'm also considering skipping token data entirely and fitting a ridge map from M10 to M0 on the 1,496 sentence pairs I already have.

I'm checking whether M0 can be recovered from M10 by a ridge map fit on the 1,496 sentence pairs, and whether the gain from adding M0 comes from token-identity information that layer 10 smoothed out. Next I'd measure how well a ridge map from M10 reconstructs M0 by in-fold R², and whether an alternative that pools layer-0 signal into the CLS token would carry it through instead.

I'm considering a different route: making the CLS token itself carry M0 information through the layers, perhaps by a small perturbation at layer 0, though that would be degraded and complicated to pool cleanly. I'm also noting that M1 with M11 reaches 97.3%, but layer 1 has the same difficulty. The same problem applies to the [SEP] token or the last token.

I'm weighing whether to make the CLS position isolated, so that non-CLS tokens stop attending to it in every layer. That would turn CLS into a free accumulator, though it changes how every token is processed and could degrade the features a little.

I'm working out how to make the CLS token stop being attended to by the other tokens in every layer, so it can act as a free accumulator for M0. The obstacle is that CLS's key at layer 0 is fixed by the embedding output, so changing it there would also change the keys of every other token.

I'm working out how to pick a direction d for the layer-0 key so that its dot product with CLS's embedding is 1 while staying near zero for every other token, since the CLS embedding may not be fully orthogonal to them.

I'm weighing whether a least-squares direction d, chosen so that d·e_cls is 1 while d·h_j stays near zero for other tokens, could isolate the CLS key. The catch is that computing layer-0 embeddings for typical tokens edges toward reimplementing the forward pass, which I'd rather avoid.

I'm weighing a layer-0 option that keeps the CLS token's features intact, but it looks complicated and risky. A simpler route would be to pair M0 with a small set of later layers, though M0 would still have to survive through those layers somehow.

I'm weighing a token-level carry, where each token keeps its own embedding in a protected subspace through the layers, but that would require perturbing the representations. I'm also considering reconstructing M0 from the deep token states, perhaps by reusing the MLM head's transform, which would make token identity recoverable at the final layer.

I'm weighing whether the decoder side of the MLM head could be replicated, but a 30,522-way vocabulary output is far too large to fit into a feed-forward block with 3,072 units. Still, M0 is essentially a bag of words, and the classes depend heavily on specific keywords like "premium" or "deductible." Deeper layers blur that identity with context, so an attention head in layer 11 that searches for keyword-like token states might recover it, though designing such a query would require token-level data I don't yet have.

I'm considering using a head in layer 11 whose query is aligned with each class direction, so attention concentrates on the tokens most indicative of that class and acts as a nonlinear max-pooling over class evidence. Each of the 8 classes would get its own such query.

I'm settling on an implementation order: first uniform pooling at layer 11 on LN(M11), fit the classifier offline, and check it with one prediction pass. The max-pool heads over class evidence are a possible refinement, but they can wait until the simpler version is checked.

I'm testing whether a ridge map from M10 to M0 gives the same gain, which would just mean a better-regularized ridge on M10, and I'm also considering an attention-based keyword readout at layer 11.

I'm wondering whether a per-token route could carry the layer-0 signal, and I'm considering a variant that carries M0 in the CLS token with a small perturbation at layer 0, since that perturbation would propagate roughly linearly through CLS's residual stream.

I'm considering whether the CLS token could carry the layer-0 signal if its perturbation were small enough to propagate roughly linearly. The concern is that the base CLS representation varies a lot from sentence to sentence, so a small added M0 component would be buried in that variation and hard for a readout to separate.

I'm considering whether the CLS attention output could overwrite the CLS state at layer 1, so that its base content is replaced by a pooled M0 projection. That would change the keys and values other tokens see, so I'm checking how much base BERT's tokens attend to CLS in the early layers.

I'm considering a cleaner way to make CLS uniform in layer 0: since its input is the fixed embedding e_cls, its query is constant, so I could adjust W_q for one head to zero out that query and get uniform attention, while leaving the other tokens' queries unchanged.

I'm checking whether a projection direction d can give CLS a constant query while leaving the other tokens' queries essentially unchanged, using a covariance-weighted choice of d so that d·h_j stays near zero for non-CLS embeddings.

I'm considering whether a head in layer 0 could be repurposed for CLS alone, since the other tokens' queries would barely shift if the projection is small. But pruning or zeroing a head would make it uniform for every token, so the head's output would reach all of them, which I want to avoid.

I'm considering using the head's value and output projections as they are, so that other tokens keep their normal outputs, and only the CLS token's pooled contribution changes. That would limit the readout to a 64-dimensional projection of M0 rather than the full signal.

I'm checking whether all twelve heads in layer 0 could be set so the CLS query is uniform, which would make CLS's attention output a full linear map of M0 while other tokens stay nearly unchanged.

I'm checking whether the CLS attention output in layer 0 can carry the full M0 signal, since making every head's CLS query uniform would turn its output into a full linear map of M0. Then CLS goes through layers 1 to 10 with the base weights, and I'm wondering whether the resulting deep CLS state C11' still holds that information.

I'm checking whether CLS states stay distinguishable from other tokens at each layer, since a per-layer linear functional would need that separation to keep the trick working. Base BERT's CLS states do look like they cluster apart from the rest, which would make the layer-by-layer version plausible.

I'm weighing whether a per-sentence query correction could make CLS attend uniformly, since its query varies with the sentence and a single rank-1 fix only cancels the average. Any leftover variation would make the attention non-uniform, and I'm wondering if a scaled-down query might be enough. Given how much complexity this adds, I'm inclined to weigh it against the small expected gain.

I'm weighing whether pooling at an earlier layer with nonlinear features could beat the layer-11 options, since M6 alone already gets about 96.5%. I'd also like to check how much token identity survives linearly in the layer 10 and 11 states, which would need token-level reads.

I'm checking whether the [CLS] token could carry the shallow M0 signal, but the same problem appears there, since M0 is not available at layer 11 for either pooling or the CLS readout. That leaves the shallow information needing to be carried by the tokens themselves, so I'm looking at whether token states could hold it.

I'm considering whether layers 2 through 10 could be made close to identity so that the layer-11 state simply reflects a shallow layer, though that would discard the deeper features I want. Splitting the tokens between roles or using only the special tokens and punctuation seems unlikely to help. I'm now wondering whether the attention patterns could supply the complementary information instead, and I'm asking whether the gain from M0 comes from it acting like an uncontextualized bag of words.

I'm considering whether a per-token nonlinear map in layer 10 could recover token identity, since its FFN sees each token after attention. A linear reconstruction from the layer-10 mean back to the layer-0 mean might be decent, so I want to test ridge regression from M10 to M0 and check classification on M10 alone.

I'm going to check how well M10 can be mapped back to M0 with a ridge regression, and then test classification using both the M10 features and that reconstruction.



The shallow mean feature M0 pairs best with the deep layers (about 97.6% with M10), while any single deep layer reaches only about 96%, so I'm checking whether M0's information can be linearly recovered from the deep features.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\nfor src in ['M10','M11']:\n    for lr in [1,10,100]:\n        def fn(tr):\n            A=stdz(F[src],tr); B=stdz(F['M0'],tr)\n            R=np.linalg.solve(A[tr].T@A[tr]+lr*np.eye(768),A[tr].T@B[tr])\n            Bh=A@R\n            return np.concatenate([A, Bh/ (Bh[tr].std(0)+1e-6)],1)/np.sqrt(2)\n        # R2 on held-out\n        tr=folds!=0; te=~tr; A=stdz(F[src],tr); B=stdz(F['M0'],tr)\n        R=np.linalg.solve(A[tr].T@A[tr]+lr*np.eye(768),A[tr].T@B[tr]); r2=1-((A[te]@R-B[te])**2).sum()/((B[te]-B[tr].mean(0))**2).sum()\n        print(src,lr,'R2',round(r2,3),[cv(fn,l) for l in [30,100,300]])"
}
```


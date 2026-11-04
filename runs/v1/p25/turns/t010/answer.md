## reasoning summary

Random features land a little higher, around 96.5 to 96.8 percent, but they're noisy. The multi-layer linear probe reaches roughly 97.5 to 97.9 percent. At this level the gap is about 45 errors versus 33, and the cross-validation standard deviation is maybe half a point. Now I'm weighing how to get features from several layers into the final CLS token.

I'm now weighing a few ways to carry CLS information through the later layers without losing it, since the residual path keeps earlier information around even as attention and FFN outputs mix in new content. One option is to repurpose a single head per later layer as a uniform pooling head that reads out a small set of discriminative directions.

I'm weighing whether uniform pooling heads in the later layers could push pooled information into CLS across layers, though that would also shift every other token's features. Making the head uniform only for CLS would need CLS-specific query features, and setting the query weights to zero makes it uniform for all tokens, so I'm not sure that's workable.

I'm checking whether CLS has a distinct direction in the hidden states that could gate the attention, but making the query uniform only for CLS looks too fiddly to get right.

I'm weighing a constant pooled vector added to every token, which would keep the CLS signal small but still mixed with its existing content, so the classifier might not separate it cleanly.

I'm checking whether some direction u can make every query's score against a marked CLS key strongly negative, so other tokens effectively ignore CLS.

I'm weighing whether a head that attends uniformly for every token could feed a sentence-level summary into CLS without swamping its existing content. The worry is that such a shared vector would perturb every other token too, and CLS signal could end up buried in the same directions where its normal content varies. Checking whether a shared gate direction could make other tokens ignore CLS would need token-level hidden states from data, which is expensive, and the expected gain of roughly 1.5 points may not justify it.

I'm wondering whether the later layers could carry multi-layer information through every token rather than just CLS, since the mean of the final layer only decodes around 95.9 percent while the concatenation reaches 97.8. The problem is that information gets mixed nonlinearly as it passes up the stack, so a simple skip that adds a projection of each earlier layer into all tokens would perturb the processing that follows.

I'm checking which layer subsets might work best, starting with {0, 6, 11} and a few alternatives like {6, 11}, {0, 11}, and {3, 7, 11}, since the earlier concatenation of [0, 6, 12] already reached about 97.5 percent.

Now I'm looking at whether the pooled vector can reach CLS alone: uniform attention for the CLS query, with other queries attending only to themselves, so the added term never touches the other tokens.

I'm considering whether the CLS query could be made to attend only to itself while other tokens ignore it, perhaps by giving CLS a value that is near zero. That would depend on whether CLS states are stable enough across sentences to serve as a reliable zero-value sink, and I'm not sure they are. I'm also weighing whether a position-dependent query set at the embedding layer could separate CLS from the other tokens more cleanly.

I'm weighing a design where one repurposed head in an early layer, like layer 6, does uniform attention to pool features, and I'm considering whether projecting that pooled vector into all tokens would be acceptable if the classifier is trained on the edited model's real features.

I'm weighing whether a fixed set of pooled projections, written into every token's residual stream, could carry multi-layer information through to the final layer. The concern is that the added vectors would be reshaped by later layer norms and feed-forward blocks, and that would perturb the processing that follows. I'm leaning toward checking whether the benefit can come from the CLS token's own base states instead.

I'm checking whether the CLS state from layer 11 could mix in mean(h11) directly, but superimposing the two in the same dimensions would lose information compared with concatenating them. Another option is to keep some heads in layer 11 for pooling and let the FFN handle the CLS path, so I'm working through how that residual update would go.

I'm noting that all heads in layer 11 see the same tokens, so with uniform attention the CLS output is just a linear function of the mean vector, one 64-dim slice per head. A non-uniform head could instead use a class-like query to pool more selectively, which would be a nonlinear pooling option.

I'm weighing whether a fixed attention query could pull class-relevant tokens into CLS, perhaps with a learned direction matched to LDA-style token features, though that would need token-level states that are expensive to collect. I'll also check which poolable layer subsets give the best concatenation, and whether a global added vector could help without disturbing the other tokens.

I'm weighing whether CLS could be isolated from other tokens' attention, perhaps by measuring how much non-CLS tokens attend to CLS in the base model across heads and layers. If that influence is small in later layers, changes to CLS would barely disturb the rest, and I could simply test an edited design and compare cross-validated accuracy.

I'm considering whether repurposing one head per layer for uniform pooling would be acceptable, since it would alter every token, not just CLS. Making the CLS query distinct enough to isolate it seems to require something I can't cleanly get, so I'm weighing whether the global perturbation from losing a single head is small enough to ignore.

I'm looking at whether the repurposed head's broadcast vector could sit in a residual subspace that downstream layers ignore, though LayerNorm's mean subtraction and scaling may disturb it.

I'm checking whether the broadcast vector could sit in a residual subspace that downstream layers mostly ignore. The read weights in later layers are full rank, so there's no exact null space, though some low-singular directions might carry little energy, which would limit the effect.

I'm weighing whether the broadcast vector could be kept out of the directions that later layers actually read, since the LayerNorm gammas would reshape it per dimension. Even so, it stays linear in the broadcast, so the pooled mean in layer 11 would still pick it up, and I'd judge the design by fitting the readout on the real edited features. I'm also still thinking about which layer combinations, like [0, 6, 11], might give the best concatenation.

Now I'm weighing whether to broadcast only the top-r PCA directions of the layer means instead of ridge class scores, since fitted scores would leak into the final classifier, and I'm checking how much accuracy that projection keeps.

I'm comparing how well mean(h11) alone does against mean(h11) combined with a low-dimensional projection of an earlier layer's mean, either LDA-7 or PCA-32.

I'm going to compare the mean of layer 11 alone against combinations with a low-dimensional projection of earlier layer means, and across the layer sets I'm considering as concatenations.



Random GELU features only lift the single-layer mean to about 96.5%, versus roughly 97.7% from multi-layer concatenation. I'm next testing which combinations of poolable layers 0–11, including low-dimensional PCA or LDA projections of earlier layers, add the most on top of mean(h11).



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef LN(X): \n    X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\nfor Ls in [[0,11],[3,11],[6,11],[8,11],[0,6,11],[0,3,6,9,11],list(range(12)),[0,6],[0,5],[2,6]]:\n    fn=lambda tr: np.concatenate([stdz(F[f'M{L}'],tr) for L in Ls],1)/np.sqrt(len(Ls))\n    print(Ls, [cv(fn,l) for l in [10,30,100,300]])\n# low-dim extras\ndef pca_proj(X,tr,r):\n    Xs=stdz(X,tr); U,S,Vt=np.linalg.svd(Xs[tr],full_matrices=False); return Xs@Vt[:r].T/ S[0]*np.sqrt(tr.sum())\nfor k in [0,6]:\n    for r in [16,64,128]:\n        fn=lambda tr: np.concatenate([stdz(F['M11'],tr), pca_proj(F[f'M{k}'],tr,r)],1)\n        print('M11+PCA',k,r,[cv(fn,l) for l in [30,100,300]])"
}
```


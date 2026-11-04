## reasoning summary

I'm checking how much accuracy the PCA-384 compression costs: about 0.4%, with the combined 384+384 landing at 97.0% against 97.4% for the full concatenation. The single M11 PCA-384 result is also a bit below its full-dimension score. I'm also noting that the 768-dim final CLS bottleneck is a real limit, and I'm wondering whether a supervised compression could beat PCA, since the final readout may only need around 8 dimensions.

I'm working out a way to have the final CLS hold just 8 supervised values in place of the full 768-dim bottleneck, using a joint ridge fit over the layer-11 features so the readout stays consistent.

I'm checking whether the final CLS can hold a 16-number summary of the readout: an 8-value pooled term from layer 11 plus an 8-value copy of the self-attention features, with the layer-norm shift and bias folded in so the argmax stays exact.

I'm settling the residual problem: scaling the output projection up by a large constant makes the 16 class-score dimensions dominate the leftover h11_cls term, so the contamination becomes negligible and the readout stays effectively exact. Once that holds, the honest CV estimate can come from fitting offline on the read features.

I'm checking whether the residual h11_cls' still carries enough of the layer-k information after the later layers transform it, since that decides how much the CLS copy heads can recover.

I'm now asking whether tapping the CLS token at a middle layer, with near-uniform attention, would preserve more of the layer-k information than tapping at the final layer. I'm also considering letting CLS self-attend in the later layers so it carries its own state forward without mixing in new token content, though its FFN would still transform it. I plan to test these variants empirically rather than assume which works best.

I'm weighing whether tapping CLS at a middle layer, with near-uniform attention at each later layer, would hold more of the layer-k signal than a single final tap. Reading the edited model's CLS states at several layers, then setting up taps at the following layers, would mean repeated passes, and I still need token states at intermediate layers, which I only have for layers 0, 6, and 11.

I'm now weighing a single tap at layer 6, since the base token states there are already on hand for the second-moment term, and the CLS state would then pass through four more layers before reaching layer 11. After that I'd read the edited model's CLS and mean hidden states at layer 11 across all 1496 examples, which takes two passes, and I'm wondering whether I could also read the CLS state at another layer.

I'm weighing a tap at layer 7 to check whether the CLS state carries the M6 information. The attention output would be roughly the same size as normal attention, so the M6 signal mixes with the base CLS state. From there I'd need to see how much survives the four remaining base layers, since CLS states at layers 8 and 11 retain only part of the earlier information.

I'm weighing a way to avoid the degradation altogether: tap at layer k, then shield the CLS state in layers k+1 through 10. Blocking the FFN for CLS alone is hard, since the same weights process the tokens, so I'd have to rely on some gating trick instead.

I'm considering a sharper variant for the CLS self-attention: shifting the FFN input for CLS alone by a large negative amount along a direction d, so its GELU units switch off and the FFN returns just its bias, while tokens get only a small perturbation since their projections onto d are tiny.

I'm making the uniform attention at each later layer add that layer's mean signal to the CLS state, so the injected information accumulates rather than acting as noise. I'm also turning off the FFN for CLS, so the layer-11 CLS state is just a layer-normalized sum of the tap and those mean contributions.

I'm considering a tied-weights version where the readout is applied to a sum of per-layer linear maps of the means, rather than to one tap, and I'm weighing whether that superposition would still classify well.

I'm settling on the simplest version first: a single tap at layer 6, with the d-trick computed from the base CLS states and base token states at that layer, and no changes in layers 7 through 10. Then I'd read the edited CLS and mean states at layer 11 and check a concatenated ridge fit against the base 96.8%.

I'm now going to test a single tap at layer 6, since I already have the base token states there for the d-direction, and I'll look at whether that holds up better than tapping at layer 4.

I'm picking the direction d by minimizing its variance over token states at layer 6 while keeping its projection onto the mean CLS state fixed at one, with a ridge term for stability. For each head, I'm computing the mean query from the mean CLS state.

I'm checking how strongly the CLS query lines up with the token queries, by computing the residual term q̄_h(d·δ) that remains after the shift. Next I want the distribution of |d·h_j| over the token states and the resulting attention entropy at the tap, which means scoring against token keys from the base L6 states.

I'm now working out the layer-11 readout: uniform pooling in some heads, and in the others a score that boosts the CLS query's match to itself through the projected d11 direction.

I'm planning to read the edited model's CLS and mean states at layer 11 with only the layer-6 tap in place, since that doesn't need the layer-11 design yet. Then I'd fit a concatenated ridge on those features and use the fit to set up the layer-11 heads: one uniform-pooling head driven by the mean readout, and one self-copy head driven by the CLS readout.

I'm working out how to make the layer-11 readout output the logits directly: one uniform-pooling head and one self-copy head both write into the first residual dimensions, with the other heads zeroed out and the FFN turned off so LayerNorm passes the sum through.

I'm checking how the pooler's tanh behaves here: with the logits spread across just 8 of the 768 dimensions, the LayerNorm output could reach about ±10 and saturate tanh, so I'm considering scaling the pooler weights down to keep the values in a usable range.

I'm now setting the layer-11 attention so that uniform pooling feeds the logits, with the classifier bias carried in through the value bias. The next concern is residual contamination from the CLS state in the first eight dimensions, so I'll scale the logits up by a large constant to keep that effect negligible in float32.

I'm checking whether the residual CLS dimensions would leak into the logit readout, since the LayerNorm sees all 768 dimensions and the self-copy idea might work without attention if the CLS state is left unswamped.

I'm checking whether the residual CLS state would still shift the logits: the LayerNorm divides everything by one per-sample σ, so the CLS and mean contributions scale together, but the mean subtraction shifts the CLS term by μ.

I'm working out how the pooler reads the B channel: with the rows of B summing to zero, the shared mean subtraction from LayerNorm cancels out, leaving the argmax intact. I'm also checking whether the residual CLS values in the channel dimensions would leak in, which would mean choosing dimensions where the layer-11 CLS state has low variance.

I'm working out how the pooler's weights would read the channel dimensions: if the layer-11 CLS state is scaled up strongly, its own values in the first eight dimensions get drowned out, so I'd drop those weights and refit the readout on the remaining dimensions.

I'm seeing that the layer-11 channel can be read out exactly: with the B rows summing to zero, the shared LayerNorm shift cancels, so the argmax of the pooled logits equals the argmax of the classifier score on the CLS state plus the mean-channel logits. That means no self-attention or swamping is needed, and the only remaining choice is the channel scale.

I'm checking how large the scale constant c should be: at c = 100 the residual contamination from the CLS state drops to about half a percent, but a bigger σ also shrinks the token-side values, so the pooler weights would need rescaling to match.

I'm checking how the residual CLS state in layer 11 combines with the mean readout, since the LayerNorm scale is shared and the mean subtraction cancels when the pooler rows sum to zero. That would make the logits exactly argmax(B·h11_cls + A·M11' + b), so I'd fit that concatenated ridge offline and implement it directly. I'm now starting to set up layer 11's first head so its value carries the classifier mapping in the first dimensions.

I'm setting the pooler rows so they take the classifier weights on the remaining dimensions plus a small scaled term in the first eight, and I'm checking that the mean subtraction still cancels.

I'm checking how the bias term and the scale constant interact here: since everything gets divided by the same LayerNorm σ, the bias stays inside the argmax, so the only remaining question is how to pick the tanh scale so the pooled values don't saturate.

I'm checking that the layer-11 readout stays sound: with the channel scaled by 100, σ lands near 7, so the pooled values stay around ±0.4 and tanh won't saturate, and the bias scales with everything else. I'll keep the pooler scale at 1 and set the classifier to the identity, with no bias.

I'm making the readout invariant to the mean shift by constraining each class row to sum to zero, which I can do by centering the fitted weights. The aim is that subtracting μ from every dimension leaves the logit ranking unchanged, so I don't need the exact LayerNorm mean in the implementation.

I'm checking that the B rows sum to zero over dims 8 onward, so the mean-shift from LayerNorm drops out of the pooled logits. Fitting on dim-centered features gives the same readout, and the channel dimensions should be chosen where the CLS state has the least variance.

I'm checking the LayerNorm step after the attention: the feed-forward branch is zeroed out, so the second LayerNorm leaves the already-normalized vector unchanged, and the final CLS state is just that vector.

Next I'll check that layer 11 does what I need on its own: uniform pooling with zeroed query and value weights, so the logits come straight from the CLS state and the mean-pooled hidden state. The plan is to start with the base C11 and M11 features, which already reach about 96.8% combined, and only then try tapping layer 6 for a richer CLS readout.

Next I'm writing the notes, then fitting a ridge readout on the base layer-11 features over all 1496 examples, with the regularization strength chosen by cross-validation. After that I'll run a prediction pass on a sample to check that accuracy comes out near the training level.

Next I'm writing up the notes, then fitting a ridge readout on the base layer-11 features across all 1496 examples. The regularization strength will come from cross-validation, and I'll drop the eight channel dimensions from the CLS block first. After that I'll run a prediction pass on a sample to check that accuracy stays near the training level.



Compressing to 384+384 PCA dims costs about 0.4%, and layer 11 can be built to take exactly the 8 class scores, so I'll set up a concat ridge on [C11, M11] with no refit or leakage, targeting about 96.8%, and record it in the notes.



## tool_use: run_python

```json
{
 "code": "open('notes.md','a').write(\"\"\"\n## Findings (session 1)\n- mean_hidden = masked mean over ALL tokens incl CLS/SEP (matches uniform attention).\n- Base features: feats.npz (M0..M12 mean, C0..C12 cls, y), folds.npy (5-fold stratified). passes 2-3.\n- CV ridge (stdz): single-layer mean ~95.6-96.5%; CLS worse. Concat shallow+deep helps: [M0,M11] 97.5, [M4,M11] 97.4, [M6,M11] 97.4, [C11,M11] 96.8.\n- M0 NOT linearly recoverable from M11 (sentence R2 .55, token-level R2 .38). No inert residual subspace (QKV min sv ~1/3 median).\n- Attention to CLS from other tokens (pass 6): enc layers 1-3 high (20-35%), layers 4-11 low (2-5%).\n- Token states for 128 texts at L0/6/11: tok128.npz (passes 4-5).\n## Exact readout design (layer 11, only CLS output matters)\n- head0 uniform (Wq=0,bq=0), value rows0..7 = A (8x768), bias b -> CLS gets A\u00b7M11+b. O writes c*(.) into 8 channel dims ch, other heads V=0.\n- LN1 gamma=1,beta=0; FFN W2=0,b2=0; LN2 gamma=1,beta=0 -> final CLS = LN(h11_cls + c*chan).\n- pooler row k = s*(B_k on non-ch dims (B_k summed=0) + e_ch[k]/c); classifier = I. => argmax(B h11cls + A M11 + b) exactly.\n- Plan: first implement with base C11/M11 (target ~96.8%), then try a layer-6 'CLS tap' (rank-1 W_q fix making CLS attention ~uniform) to inject M6 into CLS.\n\"\"\")\nF=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\nC=F['C11']; M=F['M11']\n# choose channel dims: lowest variance dims of C11\nv=C.var(0); ch=np.argsort(v)[:8]; print('channel dims',ch, v[ch].round(4), 'median var',np.median(v).round(3))\nkeep=np.setdiff1d(np.arange(768),ch)\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nCc=C[:,keep]-C[:,keep].mean(1,keepdims=True)\nfor wc in [0.5,0.7,1.0]:\n    fn=lambda tr: np.concatenate([wc*stdz(Cc,tr),stdz(M,tr)],1)\n    print('wc',wc,[cv(fn,l) for l in [100,200,300,500,800]])"
}
```


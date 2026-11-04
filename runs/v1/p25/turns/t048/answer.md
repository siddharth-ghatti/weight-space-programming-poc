## reasoning summary

I'm checking how head 0 in layer 0 behaves: its attention is nearly uniform, so zeroing its query weights and bias would make it exactly uniform with only a small change. I still want its value outputs to carry the mean of the values, and I need to figure out how to fit 8 extra stash dimensions for the A0·M0 term alongside its 64 value dimensions.

I'm weighing a way to free up room for the 8 stash dimensions: either swap 8 of head 0's 64 value rows over to carry A0·M0, accepting a small loss in its value information, or keep the value rows intact and derive the stash from a fixed 64-dimensional projection of M0 instead, which I'd need to check offline.

I'm considering freeing up the 8 residual dimensions for the stash by having layer 0's frozen outputs cover them, since the rest of the design depends on which dimensions get locked in. Choosing those dimensions is still open.

I'm now thinking about which residual dimensions to free for the stash: dimensions with nearly constant activation across tokens could be mean-ablated, folding their average contribution into the reading biases and zeroing the weight columns.

I'm noting that the stash dimensions also feed into the mean and variance used by each token's layer norm, so their values shift those statistics a little. Since the head's output is nearly the same for every token, the stash is roughly constant across tokens, and I'm wondering whether the final readout still works when those dims are included.

I'm weighing whether the stash dimensions should get a LayerNorm gain of 1 and no bias, or a constant gain matched to the typical scale of the residual stream. The unsettled question is how large σ is for these pre-LN vectors, since over 16 normalizations the stash could drift in magnitude.

I'm weighing whether the stash can be sized empirically: write it with a unit gain, then read the hidden states at each layer to see how much it drifts before deciding on any per-layer gain correction.

I'm checking whether the stash can tolerate the layer-norm rescaling: if its magnitude shrinks or grows across layers, I can absorb that by rescaling its readout weights, and float32 precision should hold unless the scale collapses far enough.

I'm wondering whether the stash magnitude will drift across the 16 layer norms, since a dimension that grows to around 30 would distort each token's normalization. I'd want to estimate the pre-norm scale from the stored hidden states before choosing a gain.

I'm weighing whether to estimate the pre-norm scale empirically rather than computing it myself, since reimplementing the forward pass is off the table. The idea is one read pass with the stash dimensions set to a unit gain and a small value, then reading the cls hidden states at every layer to see how the stash magnitude changes across the layer norms. From those per-layer ratios I could set a gain correction in each layer's second norm to keep the stash at a steady scale.

I'm checking whether the stash can stay CLS-only: the readout at layer 9 would use CLS's stash dimensions, so the per-sentence scale factor is just the product of CLS's pre-norm standard deviations across layers. CLS states look fairly similar from sentence to sentence, so that variation may be small, but I'd need to confirm it.

I'm simulating how much the stash scale factor α can wander, with multiplicative noise of 10 to 30 percent, to see whether the M0 term still gives a usable readout. I'm also considering whether a paired stash with opposite signs cancels the scale, though it doesn't, so that idea drops out.

I'm planning the readout pipeline: first the layer-0 stash edit with the uniform head and the 8 value dimensions, then reading the layer-9 CLS and mean states plus intermediate CLS states for the scale diagnostics, and finally an offline cross-validated fit on those features.

I'm laying out the rest of the pipeline: the stash edit, a few read passes, then an offline fit, which should come to roughly five or six passes out of the fifteen available. The open question is how to choose the A0 weights for the stash, since the joint ridge fit on the stashed dimensions would be in-sample and could overfit.

I'm checking whether the joint ridge fit overfits the stash dimensions, since the M0 features are in-sample and would look more reliable than they are on test data. One option is to freeze the base readout weights and add the scaled stash on top, and the other is to refit using the exact M0 features.

I'm weighing whether to fix the stash to unsupervised PCs of M0 (eight of them) instead of ridge-fitted weights, since that avoids the in-sample overfit a full joint fit would introduce.

I'm going to test whether the stash still carries the gain if I use only the top-k principal components of M0, computed without labels, instead of fitting weights on it. Three options are on the table: unsupervised PCs at several values of k, class-mean-difference projections computed within each fold, and a noise simulation for the scale factor.



Layer-0 head 0 is almost uniform, so it's a cheap head to reuse for M0 pooling. Next I'll test offline whether a low-dimensional M0 projection keeps the gain, and check robustness to per-sentence scale noise α.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64); M0=F['M0'].astype(np.float64)\ndef stdz(X,tr): m=X[tr].mean(0); s=X[tr].std(0)+1e-6; return (X-m)/s\ndef ridge_pred(Xtr,Y,lam,Xte):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0)\n    n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return (Xte-xm)@Wt+ym\ndef run(extra_fn, lams, w=1.0):\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            E=extra_fn(tr)\n            X=np.concatenate([stdz(C9,tr),stdz(M9,tr)]+([w*E] if E is not None else []),1)\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams: res[l]+=(ridge_pred(X[tr],Y,l,X[te]).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nM0c=M0-M0.mean(0); U,S,Vt=np.linalg.svd(M0c,full_matrices=False)\nfor k in [8,16,32,64]:\n    P=M0c@Vt[:k].T; P=P/P.std(0)\n    for w in [1,3]:\n        print('PC',k,'w',w, run(lambda tr: P-P[tr].mean(0), [300,1000],w))\n# class-mean (LDA-ish) within fold\ndef lda_fn(k_reg):\n    def fn(tr):\n        X=M0; mu=X[tr].mean(0); Xs=(X-mu)\n        Sw=np.cov((Xs[tr]-np.array([Xs[tr][y[tr]==c].mean(0) for c in range(8)])[y[tr]]).T)\n        Mc=np.array([Xs[tr][y[tr]==c].mean(0) for c in range(8)])\n        W=np.linalg.solve(Sw+k_reg*np.trace(Sw)/768*np.eye(768),Mc.T)  # 768x8\n        P=Xs@W; return P/P[tr].std(0)\n    return fn\nfor kr in [0.1,1.0]:\n    for w in [1,3]:\n        print('LDA reg',kr,'w',w, run(lda_fn(kr),[300,1000],w))"
}
```


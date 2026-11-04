## reasoning summary

Deep layers look fragile here: dropping the FFN width to m=768 everywhere costs the deeper features about three points (97.4 down to 94.3), so pruning that way hurts badly. That means the deep path would still need most of its FFN units, which makes it expensive. The shallow features top out around 97.1–97.3% on the full set, so I'm now looking for the cheapest configuration that can reach roughly 97.8% cross-validation, and I'm weighing a few options.

I'm comparing two cheap configurations: pruning to the first 9 layers with full pooling gives about 98.0% at 95M parameters, and the first 5 layers with a k=6 pooled head gives about 97.5% at 67M. Both are well below the 109.5M starting model, which feels like a poor trade. Since shallow features are where the size savings are, I'll look harder at improving them with cheap per-token nonlinear features in the early layers.

I'm now weighing whether the readout can work on a per-token nonlinear map instead. Since the pooling is linear, the classifier only needs a few directions in the FFN output, so I could project 3072 per-token features onto the class dimensions and mean-pool the resulting scores.

I'm now checking whether the mean of per-token nonlinear features could give a richer sentence representation than the kernel on the mean embedding. The readout is linear, so only a few projected directions matter, which makes a closed-form ridge fit on 3072-dim mean features plausible. Getting those features out of the 768-dim residual stream, though, looks awkward, since the layer norm mixes things.

I'm checking whether a per-token nonlinear feature map could be read out through mean pooling, since the pooling is linear. The catch is that LayerNorm after the FFN breaks the clean mean, so I'd need a probe setup that avoids that distortion before I could judge whether those features help.

I'm now settling on a cleaner plan: treat the model as a configurable feature extractor, read out its features, and fit a closed-form readout, using cross-validation on those features as the accuracy estimate. I'll iterate on designs with single passes over 1024 examples, then run the final two passes over the full set.

I'm sketching the design space for a two-layer setup: the first BERT layer keeps its original attention, with m FFN units, and a pool layer with 12 attention heads feeds a CLS readout, with an FFN of m units giving more capacity. At m=3072 that readout is much richer but costs about 38.6M parameters. A one-layer variant, with just the pool layer and a CLS FFN as readout, looks like it would be cheaper still.

I'm weighing whether a weighted bag-of-embeddings could reach the 97.8% target, since the layer-0 token features are non-contextual and any pooling would just be a weighted BoW. My guess is that it won't get there, though I'm not certain.

I'm looking at the deeper configurations next: keeping nine BERT layers with a full-width pool head still lands around 95M parameters, barely smaller than the start, so that path doesn't look worth it. I'm checking whether a mid-depth setup could be the best safe option, starting with the full-set numbers for the candidate designs.

I'm weighing the deeper options against the shallow ones, and they all look like poor trades on size. Since the test threshold is 97.8% and the starting model scored 98.31%, I'm considering whether pushing the shallow model up by roughly 0.7–1 point through better design is the realistic route.

I'm considering whether max-pooling over token-level class evidence might work better than the mean. Certain words point toward specific classes, like "means" for definitions, "excluded" for exclusions, or "premium" and "deductible" for their respective categories. A mean embedding dilutes these signals with sentence length, so a max-style pooling could keep the strongest cues intact.

I'm thinking about why bag-of-words struggles with these errors: negations like "not covered" are picked up as cues, but cases such as "accidental deaths... are covered" look like exclusion patterns, so the lexical signal misleads. Accepting a moderately deeper model and trimming costs elsewhere might be the more practical route.

I'm wondering if pruning the FFN with better compensation would hold up at 1536 units, since the 768-unit version degraded the deeper features badly. Even so, ten layers at that width would still come to about 72M parameters, which is still large. I'm also considering whether the test threshold can be reasoned about more directly: the earlier model scored 97.8 in CV and 98.31 on test, so the shallow kernel's roughly 97.1–97.3 CV might translate to something near 97.6–97.8 on test.

I'm considering whether the kernel could be made more discriminative by reweighting the feature dimensions, perhaps with an LDA-style whitening step before an RBF or quadratic kernel. Other angles, like CLS features from layer 1 or pooling that excludes special tokens, look unpromising or aren't available. The approach would be to test it on M0 without changing the rest of the pipeline.

I'm going to test whether projecting onto the top discriminant or PCA components, with whitening, and then applying a quadratic kernel does better than the current setup. I'll also try mixing linear and quadratic terms with a tuned weight, and a cubic kernel. The deeper-layer gains are out of reach for the shallow model, so the question is how much a stronger nonlinear readout can recover on M0 and M1.



Narrowing the FFN hurts more at depth, falling to 94.3% at M9 with top-768 units, so deep stacks need nearly full FFNs while shallow ones plateau near 97.1–97.3%. Since the shallow regime is far smaller, I'm testing whether better-conditioned kernels (LDA-, PCA-whitened, or tuned quadratic/cubic) raise its ceiling.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\ndef kcv_fn(X, transform, kfun, lam=1e-2, seeds=(0,1,2)):\n    out=[]\n    for s in seeds:\n        f=folds(y,5,s); S=np.zeros((len(y),8))\n        for i in range(5):\n            tr=f!=i; te=f==i\n            Z=transform(X, tr)\n            Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)\n            K=kfun(Z@Z[tr].T)\n            A=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[te]@A\n        out.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(out)),4)\ndef zt(X,tr):\n    return (X-X[tr].mean(0))/(X[tr].std(0)+1e-6)\ndef lda_t(shr, ncomp=None):\n    def t(X,tr):\n        mu=X[tr].mean(0); means=np.stack([X[tr][y[tr]==c].mean(0) for c in range(8)])\n        Xc=X[tr]-means[y[tr]]; Sw=Xc.T@Xc/len(Xc); Sw=(1-shr)*Sw+shr*np.trace(Sw)/Sw.shape[0]*np.eye(Sw.shape[0])\n        ev,EV=np.linalg.eigh(Sw); Wh=EV/np.sqrt(ev)\n        return (X-mu)@Wh\n    return t\ndef pca_t(k, whiten=True):\n    def t(X,tr):\n        mu=X[tr].mean(0); U,s,Vt=np.linalg.svd(X[tr]-mu, full_matrices=False)\n        Z=(X-mu)@Vt[:k].T\n        return Z/s[:k]*np.sqrt(len(tr)) if whiten else Z\n    return t\nq=lambda C: C+0.5*C**2\nfor L in [0,1]:\n    X=M(L)\n    print(L,'z quad', kcv_fn(X,zt,q))\n    for a in [0.25,1,2]:\n        print(L,'z C+aC^2',a, kcv_fn(X,zt,lambda C: C+a*C**2))\n    print(L,'z cubic', kcv_fn(X,zt,lambda C: C+0.5*C**2+C**3/6))\n    for shr in [0.1,0.3]:\n        print(L,'lda-whiten quad', shr, kcv_fn(X,lda_t(shr),q))\n    for k in [64,128,256]:\n        print(L,'pca-white',k, kcv_fn(X,pca_t(k),q), 'pca raw', kcv_fn(X,pca_t(k,False),q))\n"
}
```


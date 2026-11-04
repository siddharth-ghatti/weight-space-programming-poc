## reasoning summary

The cross-validation estimate sits around 97.5–97.7%, while the original design scored 98.31% on the test set, so the test split probably came out a bit easier or I got lucky. The final model will train on all 1,496 examples rather than the 1,197 each CV fold uses, so it should do slightly better. A design that reaches about 97.7% in CV could plausibly land at 97.8% or higher on test.

Next I'm weighing whether to check which test errors look like label noise before choosing features. I'm aiming for at least 98% CV accuracy, since the original's margin over the threshold is only about 0.5%.

I'm considering whether a slimmer model could keep the original's feature quality. Pooling only layers 0 through k-2 gives weak features, since means of early layers reach roughly 96% and I'd need something closer to layer 6 to match the original. I'm wondering if dropping or altering intermediate layers could still produce useful representations, which would need testing.

I'm sketching two single-layer pooling designs: one where all heads average the embeddings uniformly, which should land around 96%, and another where some heads use learned attention pooling toward class directions to recover more discriminative features.

I'm now weighing a max-pooling head alongside the mean: scoring each token against the class directions and keeping the strongest evidence per class, since a single keyword can get diluted in a long sentence. Mean-plus-max pooling is often a point or so better on text classification, though I can't be sure it will help here.

I'm considering whether token-level scoring is worth doing offline, since reading per-token hidden states for each example would be expensive. A simpler route is to design the model and read the final CLS vector, though each experiment then costs two passes over the data. Using a smaller subset for quick exploration would be faster, but CV on fewer samples gets noisy, so I'm weighing that trade-off.

I'm checking whether a kernel method, like RBF kernel ridge or a polynomial kernel on the layer-0 mean features, beats the linear version. If it does, a nonlinear head could approximate that gain inside a small model. I'll also try LDA-style classifiers on the same features.

I'm going to see how far bag-of-embedding features can get on their own, checking a kernel ridge with an RBF kernel and a random-feature tanh projection against the plain ridge baseline of 96.06%.



Layer combining tops out near 97.5–97.7% CV, about where the original design sits (97.66%), which scored 98.31% on test, so its CV runs a bit pessimistic. Next I'll see whether a nonlinear head on cheap layer-0 and layer-1 means can close the gap.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef krr_cv(K, y, lams, k=5, seed=0):\n    f=folds(y,k,seed); out={}\n    Y=np.eye(8)[y]\n    for lam in lams:\n        S=np.zeros((len(y),8))\n        for i in range(k):\n            tr=f!=i; te=f==i\n            Ktr=K[np.ix_(tr,tr)]; A=np.linalg.solve(Ktr+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n            S[te]=K[np.ix_(te,tr)]@A\n        out[lam]=round(float((S.argmax(1)==y).mean()),4)\n    return out\nfor name,X in [('L0',M(0)),('L0+L1',np.concatenate([M(0),M(1)],1)),('L6',M(6))]:\n    Xs=(X-X.mean(0)); \n    D2=((Xs[:,None,:]-Xs[None,:,:])**2).sum(-1) if len(X)<0 else None\n    sq=(Xs**2).sum(1); D2=sq[:,None]+sq[None,:]-2*Xs@Xs.T; med=np.median(D2)\n    for gfac in [0.5,1,2]:\n        K=np.exp(-D2/(gfac*med)); print(name,'rbf',gfac,krr_cv(K,y,(0.01,0.03,0.1,0.3)))\n    Kl=Xs@Xs.T; print(name,'lin',krr_cv(Kl,y,(1,3,10)))\n    Kc=Xs@Xs.T/ (np.sqrt(sq)[:,None]*np.sqrt(sq)[None,:]); print(name,'cos-lin', krr_cv(Kc,y,(0.01,0.03,0.1,0.3)))"
}
```


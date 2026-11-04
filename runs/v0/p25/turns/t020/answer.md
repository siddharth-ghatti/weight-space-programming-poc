## reasoning summary

With only 64 PCA dimensions the gain is small, and the full mean0 gets to 0.984, so the useful signal isn't in the top components. A supervised projection could carry more, but it would leak labels, so I'm considering a nested CV where the projection is fit inside each fold.

I'm weighing a supervised projection of mean0 into 8 dimensions, fit only on the training fold, then combining it with the other features in a ridge model. Stacking on overfit training predictions could let the combiner overweight that projection, so I want to check the result with an honest in-fold design.

The worry is that a supervised projection of mean0, fit on all the data, would look too reliable to the final combiner. Its training-set outputs are overfit, so I'd end up with a combiner that trusts it too much. I could sidestep that by writing the combination out directly, with weights picked through cross-fitting on the practice data. But the projection gets transformed by layers 1 through 10 before it reaches the CLS output, so it no longer behaves as a clean projection, and the final classifier still ends up fit on features where that projection is in-sample.

I'm weighing whether an injected label-free projection could recover some of the discriminative signal in mean0, but PCA-64 only adds about 0.3%. Using more heads for 256 dimensions might help a little, though each head costs capacity. An honest alternative is to design the projection on one half of the practice data and test on the other, even though the final model would then not use all the data.

I'm considering whether layer-10 processing could make the layer-11 hidden states more lexical, or whether earlier-layer CLS vectors like cls6, which alone reach 0.941, could add anything. The layer-11 residual stream is the only place I can read from, so those options are limited.

I'm considering whether the layer-10 feed-forward block could be modified to add per-token nonlinear features before pooling, since the layer-11 hidden states carry token identity information that the linear pooling currently can't fully exploit.

I'm weighing a supervised projection of mean0, but the question is how to estimate its value honestly. I'll check the 8-dim bow scores inside a nested CV, and if they reach around 0.983, the extra effort looks worthwhile. Before using them I still need to work out how to handle the leakage at final fit.

I'm now asking whether the injected projection could instead be carried through a known subspace at a later layer, since its transformed form at the CLS output is hard to control. Injecting it at layer 10 would only give me mean10, so that doesn't help. The final [SEP] token would run into the same transformation problem, so I'm looking at whether layers 1 through 10 can be made to preserve the injected signal in a fixed direction.

I'm considering amplifying the injected projection so it survives layers 1 through 10, though I'd be trying to stabilize the CLS state against per-token FFN and attention updates that I can't isolate to CLS alone.

Injecting a label-free projection would survive the CLS updates only partially, but BERT's residual stream keeps token identity fairly intact across layers, so a linear probe on the layer-11 CLS state could still decode it. The catch is that a supervised projection would make the CV estimate optimistic, so only a label-free one gives an honest number.

I'm going to measure the honest value of a supervised projection using nested CV. I'll fit the 8-dimensional projection of mean0 only on each training fold, using inner out-of-fold scores for the training rows and the full-fold projection for the test rows, then combine it with the layer-11 features in a ridge model. If the gain clearly beats PCA-64's +0.3%, the extra complexity is worth it.



A 64-dimensional label-free PCA summary of mean0 adds about 0.3%, versus about 0.9% for the full mean0. Next I'll test a supervised bag-of-words summary using nested cross-fitting so the estimate isn't leaky.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nm0=D['mean0'].astype(np.float64); c11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nrng=np.random.default_rng(1)\ndef oof_scores(X,yy,lam,nf=5):\n    inner=rng.integers(0,nf,len(yy)); Z=np.zeros((len(yy),8))\n    for f in range(nf):\n        t=inner!=f; Wt,b=ridge_fit(X[t],yy[t],lam); Z[~t]=X[~t]@Wt.T+b\n    return Z\nfor lam0 in [30,100]:\n  for scale in [1,3,10]:\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f\n        W0,b0=ridge_fit(m0[tr],y[tr],lam0)\n        Ztr=oof_scores(m0[tr],y[tr],lam0); Zte=m0[~tr]@W0.T+b0\n        Xtr=np.hstack([scale*Ztr,c11[tr],m11[tr]]); Xte=np.hstack([scale*Zte,c11[~tr],m11[~tr]])\n        Wt,b=ridge_fit(Xtr,y[tr],300); pred[~tr]=np.argmax(Xte@Wt.T+b,1)\n    print(lam0,scale,(pred==y).mean(),flush=True)"
}
```


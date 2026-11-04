## reasoning summary

Shallow layers aren't enough with a linear ridge on mean-pooled features; it only hits 97.8% once the deeper layers are used, which costs too much. So the next thought is that I need either a better readout or better features from the early layers. Two directions come to mind: weighting tokens non-uniformly in the attention pooling, which would need token-level data to evaluate, and putting a nonlinear readout on top of the pooler's 768 tanh units over the CLS token.

I'm weighing whether a nonlinear readout on the pooler's 768 tanh units could beat the linear ridge without the deep layers. The pooled features would be tanh(W_p h + b) with W_p random or structured, so this is a random-feature approximation of kernel ridge. I'd check it with a forward pass to read the pooled output, since the pooler sits inside the model.

I'm considering whether the FFN units in an early layer could act as per-token nonlinear feature detectors, with the attention pooling then averaging them into a bag of nonlinear token features. That might capture specific words or context-sensitive cues like negation better than a plain mean of hidden states, though the ~96% from the existing layer-0 FFN output suggests it's not an obvious win.

I'm weighing whether the second-to-last layer's FFN could produce nonlinear per-token features that the final layer's attention then averages, since the last layer's FFN only touches CLS. Token-level reads are capped at 64 ids per pass, so evaluating this directly across the full set would take about 24 passes, which is too many.

Next I'm testing whether the linear ridge gap can be closed on the layer-0 mean features alone, using a few nonlinear and normalization variants in numpy: ridge on the features plus their squares, an RBF kernel ridge on standardized means, and ridge with different normalization or whitening. I'll also check whether the mean including CLS and SEP tokens is hurting things.

I'm going to try kernel methods on the layer-0, layer-1, and layer-2 mean features, plus combinations of them, to see whether a nonlinear readout closes the gap to the linear ridge without needing the deep layers.



Ridge on uniform means needs layer 9 or deeper to reach about 97.5–97.8%, which would be costly. I'm checking first whether nonlinear readouts on shallow layers do better, using RBF kernel ridge and tanh random features as offline analysis.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\ndef krr_cv(X, y, gammas, lams, k=5):\n    f=folds(y,k,0); out={}\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\n    sq=(Z**2).sum(1); D2 = sq[:,None]+sq[None,:]-2*Z@Z.T; D2/= Z.shape[1]\n    Y=np.eye(8)[y]\n    for g in gammas:\n        K=np.exp(-g*D2)\n        for lam in lams:\n            S=np.zeros((len(y),8))\n            for i in range(k):\n                tr=f!=i; te=f==i\n                Ytr=Y[tr]-Y[tr].mean(0)\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Ytr)\n                S[te]=K[np.ix_(te,tr)]@A\n            out[(g,lam)]=round(float((S.argmax(1)==y).mean()),4)\n    return out\nfor L in [0,1,2]:\n    print(L, krr_cv(M(L), y, gammas=[0.1,0.3,1.0], lams=[0.01,0.1,1]))\nX=np.concatenate([M(0),M(1),M(2)],1); print('012', krr_cv(X,y,[0.1,0.3,1.0],[0.01,0.1,1]))\n"
}
```


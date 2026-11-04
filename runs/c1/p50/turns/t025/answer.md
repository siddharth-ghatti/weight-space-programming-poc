## reasoning summary

Whitening hurts the linear probe here, so I'm dropping it and using raw centered features scaled to unit RMS instead. The anchor approach at (2,-1) already reached 96.8%, so I'll redo the comparison with the non-whitened features and their cosine-kernel equivalents.

I'm thinking about what the pooler's input should be. Since the pooler applies an affine map before tanh, it can already do centering or projection, so the anchor features can be built in the raw-centered space with the scale set by the typical dot product size. I'll also consider GELU in place of tanh, but I'll leave that for later.



Whitening lowers linear accuracy to 95.45% from 96.3% on raw features, while the tanh anchor head reaches 96.8% with a=2, b=−1. Next I'll retune the anchor head on centered, rms-normalized raw features over a scale and shift grid and compare it against RBF kernel ridge.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef anchor_cv(X, y, a, b, k=10, seeds=(0,), lams=(0.01,0.1,1), nanc=768, cos=False):\n    res={lam:[] for lam in lams}\n    for s in seeds:\n        f=folds(y,k,s); S={lam:np.zeros((len(y),8)) for lam in lams}\n        for i in range(k):\n            tr=f!=i; te=f==i\n            mu=X[tr].mean(0); Z=X-mu; \n            if cos: Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)\n            else: Z=Z/np.sqrt((Z[tr]**2).sum(1).mean())\n            rng=np.random.RandomState(i+100*s); A=Z[tr][rng.choice(tr.sum(),nanc,replace=False)]\n            F=np.tanh(a*(Z@A.T)+b)\n            for lam in lams:\n                W,bb=ridge_fit(F[tr],y[tr],lam); S[lam][te]=F[te]@W+bb\n        for lam in lams: res[lam].append((S[lam].argmax(1)==y).mean())\n    return {lam:round(float(np.mean(v)),4) for lam,v in res.items()}\nX0=M(0)\nprint('rbf ref 10f', krr_cv(rbf(X0),y,(0.003,0.01,0.03),k=10,seeds=(0,)))\nfor cos in [False,True]:\n    for a,b in [(1,0),(2,-1),(3,-1.5),(3,-2.5),(4,-3),(6,-5)]:\n        print('cos' if cos else 'rms', a,b, anchor_cv(X0,y,a,b,cos=cos))"
}
```


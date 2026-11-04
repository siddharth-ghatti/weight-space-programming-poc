## reasoning summary

I'm checking whether the 768-unit head holds up at about 97.8% under 5-fold, against 98.2% for full kernel ridge. The features sit around a large constant offset near -1, which makes the intercept and scaled lambda awkward, so I'm going to center them over the training set and retry with lower lambda ranges. I'm also looking at whether the ridge on those 768 kernel features needs a different lambda, which is where the thought breaks off.

I'm weighing whether a penalty tied to the kernel norm, minimizing the squared residual plus λ times βᵀK_CC β, fixes the overfitting from near-unregularized least squares on 768 features and 1197 samples. I'll solve that regularized system with the exact exponential features first, then try the tanh features with K_CC built from the same tanh features at the centers.



The 768-unit head reaches about 97.8% on 5-fold, versus 98.2% for full kernel ridge, and plain ridge is poorly conditioned on these features. I'm next trying the subset-of-regressors penalty with centered features.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\nX = LNr(5*F['mean_hidden_L9']+F['cls_hidden_L9'])[:,KEEP]\ndef sor_cv(X,y,nunits=768,gamma=0.4,b0=None,lams=(1e-4,1e-3,1e-2,1e-1),seeds=(0,1,2)):\n    n=len(y); res={l:0 for l in lams}\n    for s in seeds:\n        for te in folds(n,5,s,y):\n            tr=np.setdiff1d(np.arange(n),te)\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n            rng=np.random.default_rng(s); ci=rng.choice(tr,nunits,replace=False)\n            if b0 is None: f=lambda A,B: np.exp(2*gamma*(A@B.T-1))\n            else: f=lambda A,B: np.tanh(gamma*A@B.T+b0)+1\n            H=f(Z,Z[ci]); Kcc=f(Z[ci],Z[ci]); Kcc=(Kcc+Kcc.T)/2\n            T=np.eye(8)[y[tr]]-1/8; Ht=H[tr]; G=Ht.T@Ht; R=Ht.T@T\n            for l in lams:\n                B=np.linalg.solve(G+l*Kcc+1e-8*np.trace(G)/len(G)*np.eye(len(G)),R); res[l]+=((H[te]@B).argmax(1)==y[te]).sum()\n    return {l:round(v/(n*len(seeds)),4) for l,v in res.items()}\nfor g in (0.2,0.4,0.8):\n    print('exp',g,sor_cv(X,y,gamma=g))\nfor g,b0 in ((0.4,-1.0),(0.4,-2.0),(0.8,-1.5),(0.8,-2.5)):\n    print('tanh',g,b0,sor_cv(X,y,gamma=g,b0=b0))\n"
}
```


## reasoning summary

The α=2 to 5 range holds up, with leave-one-out at 98.66%, and the linear carry term adds nothing. I'm keeping the reserved dimensions excluded from the final kernel features, so the carry heads can stay as they are without leaking anything. Next I want to check the realizable 768-unit head with 5-fold cross-validation, fitting the centers on each training fold, and compare that against the full kernel ridge regression.

I'm running 5-fold cross-validation on the 768-unit head, with the centers drawn from each training fold, and I'll compare it against the full kernel ridge baseline. I also want to try a few ways of picking the centers: random training points, class-balanced selection, and k-means centroids.



Adding the [CLS] state at α=2–5 lifts LOO to 98.66% and the linear carry scores add nothing, so the head will be a kernel on the pooled vector alone. Next I'll build the 768-unit tanh version and compare center choices and bias settings under 5-fold CV, plus a noise-sensitivity check.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\nX = LNr(5*F['mean_hidden_L9']+F['cls_hidden_L9'])[:,KEEP]\ndef head_cv(X,y,nunits=768,gamma=0.4,b0=-1.5,lams=(1e-6,1e-5,1e-4,1e-3),sel='random',seeds=(0,1,2),noise=0):\n    n=len(y); res={l:0 for l in lams}\n    for s in seeds:\n        for te in folds(n,5,s,y):\n            tr=np.setdiff1d(np.arange(n),te)\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n            rng=np.random.default_rng(s)\n            if sel=='random': ci=rng.choice(tr,nunits,replace=False)\n            else: ci=np.concatenate([rng.choice(tr[y[tr]==c],nunits//8,replace=False) for c in range(8)])\n            H=np.tanh(gamma*Z@Z[ci].T+b0)\n            if noise: H=H+noise*rng.standard_normal(H.shape)\n            Ht=np.c_[H[tr],np.ones(len(tr))]; T=np.eye(8)[y[tr]]\n            G=Ht.T@Ht; sc=np.trace(G)/G.shape[0]\n            for l in lams:\n                Wt=np.linalg.solve(G+l*sc*np.eye(G.shape[0]),Ht.T@T); res[l]+=((np.c_[H[te],np.ones(len(te))]@Wt).argmax(1)==y[te]).sum()\n    return {l:round(v/(n*len(seeds)),4) for l,v in res.items()}\nfor g in (0.2,0.4,0.8):\n    for b0 in (-0.5,-1.5,-3):\n        print(g,b0,head_cv(X,y,gamma=g,b0=b0))\nprint('balanced', head_cv(X,y,sel='bal'))\n"
}
```


## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef tanh_rf_cv(X, y, nunits, scale, lams, kind='center', k=5, seed=0, rs=0):\n    n=len(y); S={l:np.zeros((n,8)) for l in lams}\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n        rng=np.random.default_rng(rs)\n        if kind=='center':\n            Cc=Z[rng.choice(tr,nunits,replace=False)]; Hf=np.tanh(scale*Z@Cc.T)\n        else:\n            Wr=rng.standard_normal((X.shape[1],nunits)); bb=rng.uniform(-1,1,nunits); Hf=np.tanh(scale*Z@Wr+bb)\n        Ht=Hf[tr]; hm=Ht.mean(0); T=np.eye(8)[y[tr]]; tm=T.mean(0)\n        A=Ht-hm; G=A.T@A\n        for l in lams:\n            Wt=np.linalg.solve(G+l*np.eye(nunits),A.T@(T-tm)); S[l][te]=((Hf-hm)@Wt+tm)[te]\n    return {l:round((S[l].argmax(1)==y).mean(),4) for l in lams}\nfor L in [0,1,9]:\n    X=LNr(M[L])\n    for sc in [1,2,4]:\n        print(L,'center',sc,tanh_rf_cv(X,y,768,sc,(0.01,0.1,1)))\n    for sc in [0.5,1]:\n        print(L,'gauss',sc,tanh_rf_cv(X,y,768,sc,(0.01,0.1,1),kind='gauss'))\n"
}
```


## reasoning summary

I'm working out the anchor count for the fit, which could be 1008 anchors plus some linear pair units, though the carried linear term may already go through the residual. Next I need to pin down the hyperparameters for the pool fit (c, cr, lam, c0).

I'm checking the layer 11 edits: the attention output dense has 768 changed weights and a changed bias, while the value weight is untouched, so I'm guessing the change is concentrated in a single row or diagonal. I'll write a quick cross-validation of the design with a simple parameter sweep, since the exact gain c is hard to recover directly from the weights.

I'm going to sweep a few values of the gain (5, 10, 20) and see which works best, since I can't pin it down from the weights. I'll set up a 10-fold stratified cross-validation with a fixed seed, using pivoted Cholesky on the training kernel to pick the anchors.



I'm writing a reusable CV routine for the anchor-feature design and will run a small hyperparameter sweep on the full 1496 rows.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, json\nfrom poolfit import *\ndef labels_for(ids, practice, LABELS):\n    P={r['id']:r for r in practice()}\n    return np.array([LABELS.index(P[i]['label']) if isinstance(P[i]['label'],str) else P[i]['label'] for i in ids])\ndef folds(y, k, seed):\n    rng=np.random.RandomState(seed); f=np.zeros(len(y),int)\n    for c in np.unique(y):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f[idx]=np.arange(len(idx))%k\n    return f\ndef lin_feats(M0, tr):\n    X=M0[:,KEEP]; mu=X[tr].mean(0); C=X-mu; s=np.sqrt((C[tr]**2).sum(1).mean()); return C/s\ndef cv(M9, C9, M0, y, c=10.0, na=1008, lam=0.02, c0=0.5, k=10, seed=0, use_lin=True, Zgiven=None):\n    f=folds(y,k,seed); Y=np.eye(8)[y]; pred=np.zeros(len(y),int)\n    for i in range(k):\n        tr=f!=i; te=~tr\n        mu9=M9[tr].mean(0)\n        Z=pool_z(M9,C9,c,mu9) if Zgiven is None else Zgiven\n        Zt=Z[tr]; Wfull=anchor_dirs(Zt)\n        Kt=gelu(Wfull@Wfull.T*768) if False else None\n        # pivoted cholesky on gelu(cos) kernel among train\n        Kc=gelu((Zt@Wfull.T))  # cos(z_i, anchor_j) approx\n        Kc=0.5*(Kc+Kc.T)\n        piv=pivchol(Kc, na)\n        Wd=Wfull[piv]; Kaa=Kc[np.ix_(piv,piv)]\n        Phi=feats(Z,Wd)\n        Lin=lin_feats(M0,tr) if use_lin else np.zeros((len(y),0))\n        coef,b=fit(Phi[tr],Lin[tr],Y[tr],Kaa,lam,c0)\n        S=np.concatenate([Phi[te],Lin[te]],1)@coef+b\n        pred[te]=S.argmax(1)\n    return (pred==y).mean()\n'''\nopen('lib7.py','w').write(code)\nfrom lib7 import *\nd=np.load('feats_seq1024.npz')\nids=[f'p{i:04d}' for i in range(1496)]\nP=practice(); print(P[0]['id'], P[1495]['id'])\ny=labels_for(ids, practice, LABELS)\nM9,C9,M0=[d[k].astype(np.float64) for k in ['mean_hidden_L9','cls_hidden_L9','mean_hidden_L0']]\nimport time; t=time.time()\nprint(cv(M9,C9,M0,y,c=10,lam=0.02,c0=0.5), time.time()-t)\n"
}
```


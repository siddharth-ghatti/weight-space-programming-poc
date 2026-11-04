## reasoning summary

I'm going to run the cross-validation on the 1024-sample subset, comparing linear ridge on E against ridge on the pooled features. I'll also check the concat of both as a reference point even though it isn't realizable, and add a kernel model on E for comparison.



## message

Now I'll compare 5-fold CV on the 1024-item subset: linear ridge on the raw features vs ridge on pooler A's tanh features, with the RBF kernel as reference.

## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy')[:1024]; E=np.load('feats/edA_cls12.npy')[:1024]\nPa=np.load('pass_6/pooled.npy'); np.save('feats/poolA_1024.npy',Pa)\nprint('pooled stats', Pa.std(), np.mean(np.abs(Pa)>0.95))\nrng=np.random.default_rng(1); fo=np.zeros(1024,int)\nfor c in range(8):\n    idx=np.where(lab==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%5\nnp.save('feats/folds1024.npy',fo)\ndef cv(X,lam,std=True):\n    pr=np.zeros(len(lab),int)\n    for f in range(5):\n        tr=fo!=f; te=~tr\n        mu=X[tr].mean(0); sd=(X[tr].std(0)+1e-6) if std else np.ones(X.shape[1])\n        Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n        Y=np.eye(8)[lab[tr]]-1/8\n        Wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]), Xt.T@Y)\n        pr[te]=(Xe@Wt).argmax(1)\n    return round((pr==lab).mean(),4)\nprint('E lin', [cv(E,l) for l in [30,100,300]])\nprint('poolA std', [cv(Pa,l) for l in [1,3,10,30,100,300]])\nprint('poolA nostd', [cv(Pa,l,False) for l in [0.03,0.1,0.3,1,3,10]])\nmu=E.mean(0); sd=E.std(0)+1e-6; Z=(E-mu)/sd\nD=(Z**2).sum(1)[:,None]+(Z**2).sum(1)[None]-2*Z@Z.T\ndef kcv(K,lam):\n    pr=np.zeros(len(lab),int)\n    for f in range(5):\n        tr=fo!=f; te=~tr\n        A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),np.eye(8)[lab[tr]]-1/8)\n        pr[te]=(K[np.ix_(te,tr)]@A).argmax(1)\n    return round((pr==lab).mean(),4)\nprint('rbf2', [kcv(np.exp(-D/(4*768)),l) for l in [0.003,0.01,0.03]])\n"
}
```


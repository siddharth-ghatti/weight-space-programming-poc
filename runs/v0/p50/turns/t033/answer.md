## reasoning summary

The ~97.7% result has plateaued, so tuning the hyperparameters isn't helping. The multi-layer feature change is the one lever left, at about +0.5%. I'll average over three splits to check whether the gain holds up across the feature-set comparisons.



Every variant is at about 97.7% across three splits, so tuning hyperparameters has run out of room. Better input features are the only lever left, and I'm checking whether the multi-layer gain holds across those same splits.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef lnorm(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ndef mkfolds(seed):\n    rng=np.random.default_rng(seed); fo=np.zeros(len(lab),int)\n    for c in range(8):\n        idx=np.where(lab==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%5\n    return fo\nFO=[np.load('feats/folds.npy'),mkfolds(7),mkfolds(8)]\ndef run(E,lams,a=2,t=0,rho=1.0):\n    R=np.sqrt(E.shape[1]); res={l:[] for l in lams}\n    for fo in FO:\n        FS=[]\n        for f in range(5):\n            tr=fo!=f; mu=E[tr].mean(0); sd=E[tr].std(0)+1e-6\n            Z=(E-mu)/sd; C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True)\n            FS.append(np.concatenate([gelu(a*(Z@C.T)/R+t), rho*Z/R],1))\n        for lam in lams:\n            pr=np.zeros(len(lab),int)\n            for f in range(5):\n                tr=fo!=f; te=~tr; F=FS[f]; Ftr=F[tr]; m=Ftr.mean(0); Y=np.eye(8)[lab[tr]]-1/8\n                Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n                pr[te]=((F[te]-m)@Wt).argmax(1)\n            res[lam].append((pr==lab).mean())\n    return {l:round(np.mean(v),4) for l,v in res.items()}\nM={l:lnorm(np.load(f'feats/mean_L{l}.npy')) for l in range(13)}\nfor name,ls in [('11',[11]),('9',[9]),('8',[8]),('8+11',[8,11]),('0+11',[0,11]),('4+11',[4,11]),('11 + cls11',None)]:\n    E=np.concatenate([M[l] for l in ls],1) if ls else np.concatenate([M[11],lnorm(np.load('feats/cls_L11.npy'))],1)\n    print(name, run(E,[0.3,1]))\n"
}
```


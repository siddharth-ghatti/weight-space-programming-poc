## reasoning summary

I'm checking how the layer-9 features (after layer 8, with the LNr(5M+C) combination) compare to the base on the subset, where leave-one-out is 97.75 and CV is 97.1. I also want to test the final CLS output directly as features, which should roughly match LNr(5·M9+C9) with the reserved entries zeroed. Finally, I'll confirm the pooling layer is correct by comparing the final CLS against that same LNr expression computed from pass_5 index 9.



I'm comparing Experiment 3's index 9 and index 10 against base on the same 1024 sentences, and checking that the pooling layer does what it was designed to do.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y'][:1024]\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef zsall(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef evalfeat(X, y, gammas=(0.2,0.4,0.8), lams=(0.001,0.003,0.01)):\n    Z=zsall(X); sq=(Z**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y]-1/8; loo={}; cv={}\n    for g in gammas:\n        K=np.exp(-g*D2); w,V=np.linalg.eigh(K)\n        for l in lams:\n            f=w/(w+l); Hd=((V**2)*f[None]).sum(1); Yh=V@(f[:,None]*(V.T@T))\n            loo[(g,l)]=(((Yh-Hd[:,None]*T)/(1-Hd[:,None])).argmax(1)==y).mean()\n        for s in (0,1,2):\n            for te in folds(len(y),5,s,y):\n                tr=np.setdiff1d(np.arange(len(y)),te); w2,V2=np.linalg.eigh(K[np.ix_(tr,tr)]); R=V2.T@T[tr]\n                for l in lams:\n                    p=(K[np.ix_(te,tr)]@(V2@(R/(w2+l)[:,None]))).argmax(1); cv[(g,l)]=cv.get((g,l),0)+(p==y[te]).sum()\n    cv={k:v/(3*len(y)) for k,v in cv.items()}\n    return round(max(loo.values()),4), round(float(np.median(list(loo.values()))),4), round(max(cv.values()),4), round(float(np.median(list(cv.values()))),4)\nprint('LOO best, LOO med, CV best, CV med')\nprint('base M9 mix', evalfeat(LNr(5*F['mean_hidden_L9'][:1024]+F['cls_hidden_L9'][:1024])[:,KEEP],y))\nM9p=np.load(WORK/'pass_5/mean_hidden_L9.npy'); C9p=np.load(WORK/'pass_5/cls_hidden_L9.npy'); X10=np.load(WORK/'pass_5/cls_hidden_L10.npy')\nprint('m2560 idx9 mix', evalfeat(LNr(5*M9p+C9p)[:,KEEP],y))\nprint('m2560 final CLS', evalfeat(X10[:,KEEP],y))\nfor L in [1,3,5,7,9]:\n    Xb=F[f'mean_hidden_L{L}'][:1024]; Xp=np.load(WORK/f'pass_5/mean_hidden_L{L}.npy')\n    d=Xp-Xb; print(L,'shift',round(float(np.linalg.norm(d.mean(0))),2),'resid',round(float(np.linalg.norm(d-d.mean(0),axis=1).mean()),2),'base centered',round(float(np.linalg.norm(Xb-Xb.mean(0),axis=1).mean()),2))\n# pooling check\npred=LNr(5*M9p+C9p); pred[:,RES]=0; pred=LNr(pred); pred[:,RES]=0\nprint('final CLS vs designed approx: max abs diff', np.abs(pred[:,KEEP]-X10[:,KEEP]).max(), 'res dims', np.abs(X10[:,RES]).max())\n"
}
```


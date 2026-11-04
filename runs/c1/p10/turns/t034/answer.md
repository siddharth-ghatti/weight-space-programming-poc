## reasoning summary

I'm setting up an evaluation on the 1024 subset: KRR with 5-fold CV (3 seeds) and LOO, comparing the base M5 against pass_4 index 5, which should match exactly.

I'm also mixing the CLS token into the representation, with an LayerNorm over the combined 5M plus C vector.



## message

Evaluating Experiment 2 on the same 1024 sentences: kernel ridge (5-fold × 3 seeds, plus leave-one-out) at each depth of the dropped-layer network, compared with base depths.

## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); yall=F['y']; y=yall[:1024]\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef zsall(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef evalfeat(X, y, gammas=(0.2,0.4,0.8), lams=(0.001,0.003,0.01)):\n    # LOO\n    Z=zsall(X); sq=(Z**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y]-1/8; loo={}\n    for g in gammas:\n        K=np.exp(-g*D2); w,V=np.linalg.eigh(K)\n        for l in lams:\n            f=w/(w+l); Hd=((V**2)*f[None]).sum(1); Yh=V@(f[:,None]*(V.T@T))\n            loo[(g,l)]=(((Yh-Hd[:,None]*T)/(1-Hd[:,None])).argmax(1)==y).mean()\n    # 5-fold 3 seeds (standardize per fold approx by global; fine)\n    cv={}\n    for s in (0,1,2):\n        for te in folds(len(y),5,s,y):\n            tr=np.setdiff1d(np.arange(len(y)),te)\n            for g in gammas:\n                K=np.exp(-g*D2); Ktr=K[np.ix_(tr,tr)]; w,V=np.linalg.eigh(Ktr); R=V.T@T[tr]\n                for l in lams:\n                    p=(K[np.ix_(te,tr)]@(V@(R/(w+l)[:,None]))).argmax(1); cv[(g,l)]=cv.get((g,l),0)+(p==y[te]).sum()\n    cv={k:v/(3*len(y)) for k,v in cv.items()}\n    return round(max(loo.values()),4), round(float(np.median(list(loo.values()))),4), round(max(cv.values()),4), round(float(np.median(list(cv.values()))),4)\ndef feat(Mx,Cx): return LNr(5*Mx+Cx)[:,KEEP]\nprint('format: LOO best, LOO median, CV best, CV median')\nfor L in [5,8,9,10,11]:\n    print('base M%d'%L, evalfeat(feat(F[f'mean_hidden_L{L}'][:1024],F[f'cls_hidden_L{L}'][:1024]),y))\nlm={5:'after 4',6:'after 6 (skip5)',7:'after 8 (skip5,7)',8:'after 9',9:'after 10'}\nfor i in [5,6,7,8,9]:\n    Mx=np.load(WORK/f'pass_4/mean_hidden_L{i}.npy'); Cx=np.load(WORK/f'pass_4/cls_hidden_L{i}.npy')\n    print('drop57 idx',i,lm[i], evalfeat(feat(Mx,Cx),y))\nprint('check idx5 equals base M5:', np.abs(np.load(WORK/'pass_4/mean_hidden_L5.npy')-F['mean_hidden_L5'][:1024]).max())\n"
}
```


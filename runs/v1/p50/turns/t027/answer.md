## reasoning summary

I'm assembling the prep features, pulling P2m, P6m, and P10m as non-R, plus P12c as non-R, and checking that the R coordinates are zero. Next I'll run the joint ridge cross-validation over those exact features at several lambda values, adding the P11 mean as well.



## message

Assembling the prepared-model features. I'll check that R is exactly zero and that m2 matches the base, then run joint ridge CV on the real prepared features.

## tool_use: run_python

```json
{
 "code": "lab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nR = [601, 222, 88, 83, 444, 506, 254, 495]; keep=np.setdiff1d(np.arange(768),R)\ndef cat(kind,L): return np.concatenate([np.load(WORK/f'pass_6/{kind}_hidden_L{L}.npy'),np.load(WORK/f'pass_7/{kind}_hidden_L{L}.npy')])\nF={}\nfor L in [2,6,10,11,12]:\n    F[f'm{L}']=cat('mean',L); F[f'c{L}']=cat('cls',L)\nnp.savez(WORK/'prep_feats.npz',**F)\nprint('R max abs m10, c12:', np.abs(F['m10'][:,R]).max(), np.abs(F['c12'][:,R]).max())\nMb=np.load(WORK/'base_mean.npy'); print('m2 nonR diff vs base', np.abs(F['m2'][:,keep]-Mb[2][:,keep]).max())\ndef oof(X,lams):\n    out={}; Ks=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X[tr]-mu)/sd; Ze=(X[te]-mu)/sd\n        Ks.append((Z@Z.T, Ze@Z.T, tr, te))\n    for lam in lams:\n        o=np.zeros((N,8))\n        for K,Ke,tr,te in Ks:\n            al=np.linalg.solve(K+lam*np.eye(len(K)),Y[tr]-Y[tr].mean(0)); o[te]=Ke@al\n        out[lam]=round((o.argmax(1)==lab).mean(),4)\n    return out\nfor name,parts in {'c12':['c12'],'m2,m6,m10,c12':['m2','m6','m10','c12'],'m2,m6,m10':['m2','m6','m10'],'m2,c12':['m2','c12'],'m2,m6,c12':['m2','m6','c12'],'m2,m10,c12':['m2','m10','c12']}.items():\n    X=np.concatenate([F[p][:,keep] for p in parts],1)\n    print(name, oof(X,[100,200,300,500,1000]), flush=True)\n"
}
```


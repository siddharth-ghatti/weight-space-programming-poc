## reasoning summary

I'm looking at whether the gain depends on combining early and late layers, since the early layers seem to need the full 768 dimensions rather than a PCA reduction. I'm also considering a supervised projection, or stacking the two ridge models' logits, and I want to test a late-fusion sum of separately fit ridge logits.

I'm checking whether averaging separately fit ridge models on the early and late layers gets close to the 97.5% from the combined setup. If it does, I'd only need to carry eight logit values from the early layer forward, and I'll run that late-fusion test within each fold.



The gain needs distant layers, so an unsupervised low-rank m0 won't do; I'm testing whether late fusion of separately fitted ridge logits from m0 and m11 can get by with only 8 numbers from layer 0.



## tool_use: run_python

```json
{
 "code": "lab = np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nM=np.load(WORK/'base_mean.npy')\ndef fit_ridge(Xt,Yt,lam):\n    mu=Xt.mean(0); sd=Xt.std(0)+1e-6; Z=(Xt-mu)/sd\n    W=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Yt-Yt.mean(0)))\n    return lambda X: ((X-mu)/sd)@W\ndef fusion_cv(layers, lams, wts=None):\n    S=np.zeros((N,8)); per=[np.zeros((N,8)) for _ in layers]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        for i,(l,lam) in enumerate(zip(layers,lams)):\n            g=fit_ridge(M[l][tr],Y[tr],lam); per[i][te]=g(M[l][te])\n    out={}\n    for w in (wts or [None]):\n        ww = w if w is not None else [1]*len(layers)\n        S=sum(wi*p for wi,p in zip(ww,per))\n        out[str(w)]=round((S.argmax(1)==lab).mean(),4)\n    return out, [round((p.argmax(1)==lab).mean(),4) for p in per]\nprint(fusion_cv([0,11],[300,100],[[1,1],[1,2],[2,1]]))\nprint(fusion_cv([0,6,11],[300,100,100]))\nprint(fusion_cv([0,12],[300,100]))\nprint(fusion_cv([0,4,8,11],[300,100,100,100]))\n"
}
```


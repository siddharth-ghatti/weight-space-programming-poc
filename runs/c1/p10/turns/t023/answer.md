## reasoning summary

I'm now computing cross-validated accuracy on the 1024-example subset, for both the base and pruned models at each depth, using kernel ridge regression over three seeds. I'll also run a linear baseline for comparison.



## message

Now the paired comparison on the same 1024 sentences: kernel CV per depth, base vs pruned at m=1024.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); yall=F['y']; y=yall[:1024]\ndef zs(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef krr_multi(blocks, y, gammas=(0.2,0.4,0.8), lams=(0.003,0.03), seeds=(0,1,2), k=5):\n    n=len(y); res={}\n    for s in seeds:\n        for te in folds(n,k,s,y):\n            tr=np.setdiff1d(np.arange(n),te)\n            Z=np.concatenate([zs(X,tr) for X in blocks],1)/np.sqrt(len(blocks)); sq=(Z**2).sum(1)\n            D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y[tr]]-1/8\n            for g in gammas:\n                K=np.exp(-g*D2); w,V=np.linalg.eigh(K[np.ix_(tr,tr)]); R=V.T@T; Kte=K[np.ix_(te,tr)]\n                for l in lams:\n                    pred=(Kte@(V@(R/(w+l)[:,None]))).argmax(1); res.setdefault((g,l),0); res[(g,l)]+=(pred==y[te]).sum()\n    return {k_:round(v/(n*len(seeds)),4) for k_,v in res.items()}\ndef best(r): return max(r.values())\nrows=[]\nfor L in range(13):\n    Xb=F[f'mean_hidden_L{L}'][:1024][:,KEEP]; Xp=np.load(WORK/f'pass_3/mean_hidden_L{L}.npy')[:,KEEP]\n    rb=krr_multi([Xb],y); rp=krr_multi([Xp],y)\n    print(L,'base',best(rb),'pruned1024',best(rp), 'rel change', round(float(np.linalg.norm(Xp-Xb)/np.linalg.norm(Xb)),3))\n"
}
```


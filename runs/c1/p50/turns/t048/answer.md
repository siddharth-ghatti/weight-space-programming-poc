## reasoning summary

The leakage explanation fits: the start model's features carry ridge scores fit on all data, so Layer 12 at lam=1 hitting 100% is suspicious rather than a real result. The reserved dimensions hold class evidence from the full dataset, and mean_hidden at Layer 6 and above already contains those carried scores. I'm going to exclude the reserved dimensions and rerun with only the kept dimensions for Layer 3 onward. The kernel run with very low lambda also failed oddly, and I still need to look into that.

I'm going to drop the reserved dimensions and keep only the kept ones for every layer, since the leaky ridge carry-channels are what's inflating Layer 12. For the kernel, the failures at low lambda look numerical, and the kernel I used isn't guaranteed to be positive definite. I'll switch to an RBF kernel on the normalized vectors, which is.



The 100% linear CV at layer 12 is leakage from the reserved dims, so I'm rerunning the scan on the 760 non-reserved dims with an RBF kernel on cosine-normalized vectors instead.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\ndef cosn(X):\n    Xc=X-X.mean(0); Xc=Xc/Xc.std(0).clip(1e-3); return Xc/np.linalg.norm(Xc,axis=1,keepdims=True)\ndef cosn0(X):\n    Xc=X-X.mean(0); return Xc/np.linalg.norm(Xc,axis=1,keepdims=True)\nfor l in [0,1,2,3,4,6,8,12]:\n    X=B[f'mean_hidden_L{l}'][:,KEEP].astype(np.float64)\n    lin=cv_acc(X,y,lams=(10,30,100,300),k=10,std=True)\n    out=[]\n    for nm,f in [('std',cosn),('raw',cosn0)]:\n        Z=f(X); S=Z@Z.T\n        for g in [2,5]:\n            K=np.exp(g*(S-1))\n            out.append((nm,g,krr_cv(K,y,lams=(0.01,0.03,0.1),k=10,seeds=(0,))))\n    print(l, {k:round(v,4) for k,v in lin.items()}, out)\n"
}
```


## reasoning summary

I'm checking the kernel distance math: with unit-normalized vectors, the squared distance comes out to roughly 2(1 − cos), which changes what gamma means here.

The kernel comes out to about exp(0.6(cos−1)), so gamma in my Nyström setup should be around 0.6, not 5 to 20. The kernel is very wide, so I'll try gamma values of 0.3, 0.6, 1, and 2, and keep the regularization lam tiny since K is order one. A wide RBF like this behaves close to a polynomial kernel, with its quadratic terms carrying most of the nonlinearity.



These results are worse because I mis-scaled gamma: the kernel is too wide, so I'm rerunning with gamma between 0.3 and 2 and also testing a quadratic kernel.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\nrng=np.random.RandomState(0)\ndef nys_cv(X, gamma, lams, ncent=768, seeds=(0,1), lin=False):\n    out={l:[] for l in lams}\n    for s in seeds:\n        f=folds(y,5,s); S={l:np.zeros((len(y),8)) for l in lams}\n        for i in range(5):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n            Z=(X-mu)/sd; Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)\n            cidx=[]\n            for c in range(8):\n                ic=tr[y[tr]==c]; cidx+=list(rng.choice(ic, min(ncent//8,len(ic)), replace=False))\n            C=Z[cidx]; F=np.exp(gamma*(Z@C.T-1))\n            if lin: F=np.concatenate([F, Z],1)\n            Ftr=F[tr]; fm=Ftr.mean(0); Ftr=Ftr-fm; Fte=F[te]-fm\n            G=Ftr.T@Ftr; B=Ftr.T@(Y[tr]-Y[tr].mean(0))\n            for l in lams: S[l][te]=Fte@np.linalg.solve(G+l*np.eye(G.shape[0]),B)\n        for l in lams: out[l].append((S[l].argmax(1)==y).mean())\n    return {l:round(float(np.mean(v)),4) for l,v in out.items()}\ndef kern_cv(X, kfun, lams, seeds=(0,1)):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)\n    K=kfun(Z@Z.T); out={l:[] for l in lams}\n    for s in seeds:\n        f=folds(y,5,s); S={l:np.zeros((len(y),8)) for l in lams}\n        for i in range(5):\n            tr=f!=i; te=f==i\n            for l in lams:\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+l*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[l][te]=K[np.ix_(te,tr)]@A\n        for l in lams: out[l].append((S[l].argmax(1)==y).mean())\n    return {l:round(float(np.mean(v)),4) for l,v in out.items()}\nX=M(1)\nfor g in [0.6,1,2,4]:\n    print('exp-cos kernel', g, kern_cv(X, lambda C: np.exp(g*(C-1)), [1e-4,1e-3,1e-2]))\nprint('lin', kern_cv(X, lambda C: C, [1e-2,1e-1,1]))\nprint('quad', kern_cv(X, lambda C: C+0.5*C**2, [1e-3,1e-2,1e-1]))\nprint('quadonly', kern_cv(X, lambda C: C**2, [1e-3,1e-2,1e-1]))\nfor g in [0.6,1,2]:\n    print('nys', g, nys_cv(X, g, [1e-5,1e-4,1e-3]))\n"
}
```


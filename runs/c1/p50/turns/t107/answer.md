## reasoning summary

I'm checking whether the ridge penalty on the anchor and linear features is too strong at lam=0.1, since that gave 98.2 with a=1, b=0. I'm thinking a smaller lambda, around 0.01 or 0.001, and separate feature scaling may be needed, since the GELU features sit roughly between 0 and 0.84 across 1000 dimensions. I also want to look at the dual kernel ridge view of the same problem.

I'm working out the dual view: ridge on the features is equivalent to kernel ridge regression with the kernel built from the GELU features. I'm also planning to scan a smaller lambda, a few values of a and b, and possibly more than one unit per anchor, since m is limited.



## message

Anchor features need much less regularization. Scanning smaller λ and the linear-term weight:

## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nZ9=cn(d['mean_hidden_L9']); C0=cen(d['mean_hidden_L0'])\ndef anchor_cv(a, b, nanch=1000, lam=1e-2, wl=1.0, seeds=(0,1), k=10):\n    accs=[]\n    for s in seeds:\n        f=folds(y,k,s); S=np.zeros((N,8)); rng=np.random.RandomState(s)\n        for i in range(k):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            anc=rng.choice(tr, min(nanch,len(tr)), replace=False)\n            Phi=gelu(a*(Z9@Z9[anc].T)+b)\n            F=np.hstack([Phi, wl*C0]) if wl>0 else Phi\n            mu=F[tr].mean(0); Fc=F-mu\n            Wt=np.linalg.solve(Fc[tr].T@Fc[tr]+lam*np.eye(F.shape[1]), Fc[tr].T@(Y[tr]-Y[tr].mean(0)))\n            S[te]=Fc[te]@Wt\n        accs.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(accs)),4)\nfor a,b in [(1,0),(2,0),(2,-0.5),(1,0.5)]:\n    for wl in [0.0,1.0,3.0]:\n        print(a,b,'wl',wl,[anchor_cv(a,b,lam=l,wl=wl) for l in [0.001,0.01,0.03]])"
}
```


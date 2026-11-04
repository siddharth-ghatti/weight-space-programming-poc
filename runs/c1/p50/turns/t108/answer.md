## reasoning summary

I'm checking why a single λ in the primal ridge doesn't line up with the dual KRR setup, since the linear block's features are scaled very differently.

I'm normalizing the anchor features Φ so their scale roughly matches the linear block, since their diagonal sums run into the hundreds. Then I'll try λ values of 0.001, 0.003, and 0.01.



The two feature blocks are on very different scales, so one λ would over- or under-regularize them. I'm normalizing each block to unit mean squared norm, as the kernel formulation does.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nZ9=cn(d['mean_hidden_L9']); C0=cen(d['mean_hidden_L0'])\ndef anchor_cv(a, b, nanch=1000, lam=1e-2, c0=0.5, seeds=(0,1), k=10, Z=Z9, C=C0, yv=y):\n    accs=[]; n=len(yv); Yv=np.eye(8)[yv]\n    for s in seeds:\n        f=folds(yv,k,s); S=np.zeros((n,8)); rng=np.random.RandomState(s)\n        for i in range(k):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            anc=rng.choice(tr, min(nanch,len(tr)), replace=False)\n            Phi=gelu(a*(Z@Z[anc].T)+b); Phi=Phi-Phi[tr].mean(0); Phi/=np.sqrt((Phi[tr]**2).sum(1).mean())\n            F=np.hstack([Phi, np.sqrt(c0)*(C-C[tr].mean(0))]) if c0>0 else Phi\n            Wt=np.linalg.solve(F[tr].T@F[tr]+lam*np.eye(F.shape[1]), F[tr].T@(Yv[tr]-Yv[tr].mean(0)))\n            S[te]=F[te]@Wt\n        accs.append((S.argmax(1)==yv).mean())\n    return round(float(np.mean(accs)),4)\nfor a,b in [(1,0),(2,0),(3,-1),(4,-2)]:\n    for c0 in [0.0,0.5]:\n        print(a,b,'c0',c0,[anchor_cv(a,b,lam=l,c0=c0) for l in [0.001,0.003,0.01]])"
}
```


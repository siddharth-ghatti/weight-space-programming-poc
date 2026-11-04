## reasoning summary

I'm treating cls0 as a constant, since the CLS embedding at position 0 never changes. That means I can skip its zero-variance features and handle them through the trace==0 case. I'll also save the helper module to a file so it can be reused.



## message

Layer-0 CLS is constant (singular). I'll save the helper functions to a module file for reuse and skip degenerate features.

## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np\ndef lda_fit(X, y, alpha, ncls=8):\n    mu = X.mean(0); Xc = X - mu\n    means = np.stack([Xc[y==c].mean(0) for c in range(ncls)])\n    R = Xc - means[y]\n    S = R.T@R/len(y)\n    tr = np.trace(S)/S.shape[0]\n    S = (1-alpha)*S + alpha*(tr+1e-8)*np.eye(S.shape[0])\n    Wt = np.linalg.solve(S, means.T).T\n    b = -0.5*np.sum(Wt*means,1) - Wt@mu\n    return Wt, b\ndef ridge_fit(X,y,lam,ncls=8):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\n    T=np.eye(ncls)[y]-1/ncls\n    A=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@T)\n    Wt=(A/sd[:,None]).T; b=T.mean(0)+1/ncls - Wt@mu\n    return Wt,b\ndef cv(X, y, folds, fit, p):\n    pred=np.zeros(len(y),int)\n    for f in range(folds.max()+1):\n        tr=folds!=f; Wt,b=fit(X[tr],y[tr],p); pred[~tr]=np.argmax(X[~tr]@Wt.T+b,1)\n    return (pred==y).mean(), pred\ndef macro_f1(y,pred,ncls=8):\n    f=[]\n    for c in range(ncls):\n        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\n'''\nopen('helpers.py','w').write(code)\nimport sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nfor k in [f'{t}{L}' for L in range(1,13) for t in ['cls','mean']]+['mean0','pooled']:\n    X=D[k].astype(np.float64)\n    r=[cv(X,y,folds,lda_fit,a)[0] for a in [0.1,0.3,0.6]]\n    print(k, ' '.join(f'{v:.3f}' for v in r), flush=True)"
}
```


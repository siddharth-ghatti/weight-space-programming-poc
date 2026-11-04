## reasoning summary

I'm moving on to cross-validated ridge regression now, and I'll put the helper code into a reusable module called `cvlib.py` in the WORK directory.



## message

Writing a reusable CV ridge helper (`cvlib.py`) and scoring each layer.

## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np\ndef folds(y, k=5, seed=0):\n    rng = np.random.RandomState(seed); f = np.zeros(len(y), int)\n    for c in np.unique(y):\n        idx = np.where(y==c)[0]; rng.shuffle(idx)\n        f[idx] = np.arange(len(idx)) % k\n    return f\ndef ridge_fit(X, y, lam, nc=8):\n    mu = X.mean(0); sd = X.std(0)+1e-6\n    Z = (X-mu)/sd\n    Y = np.eye(nc)[y]; Y = Y - Y.mean(0)\n    n,d = Z.shape\n    if d <= n:\n        Wt = np.linalg.solve(Z.T@Z + lam*np.eye(d), Z.T@Y)\n    else:\n        Wt = Z.T @ np.linalg.solve(Z@Z.T + lam*np.eye(n), Y)\n    return mu, sd, Wt\ndef ridge_pred(model, X):\n    mu, sd, Wt = model\n    return ((X-mu)/sd) @ Wt\ndef cv_acc(X, y, lams=(10,30,100,300,1000,3000), k=5, seeds=(0,), ret_scores=False):\n    out = {}\n    for lam in lams:\n        accs=[]\n        for s in seeds:\n            f = folds(y,k,s); S = np.zeros((len(y),8))\n            for i in range(k):\n                tr = f!=i; te = f==i\n                S[te] = ridge_pred(ridge_fit(X[tr], y[tr], lam), X[te])\n            accs.append((S.argmax(1)==y).mean())\n        out[lam] = float(np.mean(accs))\n    return out\n'''\n(WORK/'cvlib.py').write_text(code)\nimport sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nfor L in range(13):\n    a = cv_acc(R[f'mean_hidden_L{L}'][:,keep], y)\n    b = cv_acc(R[f'cls_hidden_L{L}'][:,keep], y, lams=(100,1000))\n    print(L, 'mean', {k:round(v,4) for k,v in a.items()}, 'cls', {k:round(v,4) for k,v in b.items()})\n"
}
```


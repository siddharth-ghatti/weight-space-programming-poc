## reasoning summary

I'm moving on to cross-validation for the ridge classifier, with one-hot targets centered and features standardized. With n=1496 and d=760, a primal solve is fine, so I'll put the helper code into a `lib.py` file in WORK for reuse.



## message

I'll write a small reusable CV helper (`lib.py`) and score each layer's mean and [CLS] features at a few ridge penalties.

## tool_use: run_python

```json
{
 "code": "lib = r'''\nimport numpy as np\nRES = np.array([83,88,222,254,444,495,506,601])\nKEEP = np.setdiff1d(np.arange(768), RES)\ndef folds(n, k=5, seed=0, y=None):\n    rng = np.random.default_rng(seed)\n    idx = np.arange(n)\n    if y is None:\n        p = rng.permutation(n); return [p[i::k] for i in range(k)]\n    out=[[] for _ in range(k)]\n    for c in np.unique(y):\n        ii = rng.permutation(idx[y==c])\n        for j,i in enumerate(ii): out[j%k].append(i)\n    return [np.array(sorted(o)) for o in out]\ndef ridge_fit(X, y, lam, C=8, standardize=True):\n    mu = X.mean(0); sd = X.std(0)+1e-6 if standardize else np.ones(X.shape[1])\n    Z = (X-mu)/sd\n    T = np.eye(C)[y]; T = T - T.mean(0)\n    A = Z.T@Z + lam*np.eye(Z.shape[1])\n    Wt = np.linalg.solve(A, Z.T@T)\n    # back to raw: score = ((x-mu)/sd) @ Wt + tmean\n    Wraw = Wt/sd[:,None]; b = -mu@Wraw + np.eye(C)[y].mean(0)\n    return Wraw, b\ndef cv_acc(X, y, lam, k=5, seeds=(0,), standardize=True, return_scores=False):\n    accs=[]; S=np.zeros((len(y),8))\n    for s in seeds:\n        for te in folds(len(y),k,s,y):\n            tr = np.setdiff1d(np.arange(len(y)), te)\n            Wr,b = ridge_fit(X[tr], y[tr], lam, standardize=standardize)\n            S[te] = X[te]@Wr+b\n        accs.append((S.argmax(1)==y).mean())\n    return (np.mean(accs), S) if return_scores else np.mean(accs)\n'''\n(WORK/'lib.py').write_text(lib)\nimport sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nfor L in range(13):\n    Xm = F[f'mean_hidden_L{L}'][:,KEEP]; Xc = F[f'cls_hidden_L{L}'][:,KEEP]\n    r = [cv_acc(Xm,y,lam) for lam in (30,100,300,1000)]\n    rc = [cv_acc(Xc,y,lam) for lam in (100,1000)]\n    print(L, 'mean', np.round(r,4), 'cls', np.round(rc,4))\n"
}
```


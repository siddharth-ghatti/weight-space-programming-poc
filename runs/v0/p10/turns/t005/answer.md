## reasoning summary

I'm checking whether 1024 samples are enough to pick the best layer by running CV on them now, instead of reading the other 472 first. Since the final fit needs edited-model features for all 1496 samples anyway, reading the rest with the base model may not be needed. I'm starting to set up a CV run with a ridge classifier using one-hot targets.

I'm also running LDA with shrinkage on each layer's CLS and mean-pooled features.



## message

Pass 2 worked. Before using another pass, I'll run 5-fold cross-validation with closed-form ridge/LDA on these 1,024 items to see which layer and pooling (CLS vs mean) gives the best features.

## tool_use: run_python

```json
{
 "code": "\nimport json\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny_all = np.array([lab2i[p['label']] for p in P])\ny = y_all[:1024]\nrng = np.random.default_rng(0); perm = rng.permutation(1024); folds = np.array_split(perm, 5)\n\ndef ridge_cv(X, y, lams, std=True):\n    res = {}\n    for lam in lams:\n        correct = 0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(len(y)), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else 1\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]*2-1\n            A = Xt.T@Xt + lam*np.eye(X.shape[1])\n            Wt = np.linalg.solve(A, Xt.T@(Y-Y.mean(0)))\n            pred = (Xv@Wt + Y.mean(0)).argmax(1)\n            correct += (pred==y[f]).sum()\n        res[lam] = correct/len(y)\n    return res\n\nlams = [1, 10, 100, 1000]\nfor kind in ['cls','mean']:\n    for Lyr in range(13):\n        X = np.load(f'pass_2/{kind}_hidden_L{Lyr}.npy').astype(np.float64)\n        r = ridge_cv(X, y, lams)\n        print(kind, Lyr, {k: round(v,3) for k,v in r.items()})\n"
}
```


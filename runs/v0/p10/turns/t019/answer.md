## reasoning summary

I'm going to check that the edited model's CLS hidden state at layer 12 roughly matches the layer norm of the mean hidden state from layer 8. I'll compare that against the Zn value computed from the base model's mean layer-8 hidden states, then run cross-validated ridge regression on the pooled data.



## message

Pass 4 done. I'll confirm the edited final [CLS] equals the normalised layer-8 mean, then cross-validate ridge on the actual `pooled` outputs.

## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nC12 = np.load('pass_4/cls_hidden_L12.npy').astype(np.float64); PO = np.load('pass_4/pooled.npy').astype(np.float64)\nM8 = np.load('pass_2/mean_hidden_L8.npy').astype(np.float64)\nZn = (M8 - M8.mean(1,keepdims=True))/M8.std(1,keepdims=True)\nprint('max diff cls12 vs LN(mean8):', np.abs(C12-Zn).max(), ' pooled range', PO.min(), PO.max())\ndef cv_acc(X, y, lam, reps=4, std=True):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else np.ones(d)\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nprint('pooled std', [cv_acc(PO,y,lam) for lam in [100,200,400]])\nprint('pooled nostd (scaled by 1/0.1)', [cv_acc(PO/0.1,y,lam,std=False) for lam in [100,200,400]])\n"
}
```


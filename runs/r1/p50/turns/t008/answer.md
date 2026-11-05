## reasoning summary

I'm checking how well the new options agree with the original head, and weighing two refit approaches under cross-validation: refitting the output projection on pooled features to match base logits, or a ridge regression on the CLS features against those same logits.



## message

Now evaluating the drop-layer-4 model: agreement with the original head, and with closed-form head refits (out_proj refit on the pooled vector, or full linear refit on `<s>`), both cross-validated.

## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, json\ny = np.load(WORK/'y.npy'); BL = np.load(WORK/'base_logits.npy'); bp = BL.argmax(1); folds = np.load(WORK/'folds.npy')\ndef f1macro(y, p, K=4):\n    f=[]\n    for k in range(K):\n        tp=((p==k)&(y==k)).sum(); fp=((p==k)&(y!=k)).sum(); fn=((p!=k)&(y==k)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\ndef ridge_fit(X, T, lam, std=True):\n    mu = X.mean(0); sd = (X.std(0)+1e-6) if std else np.ones(X.shape[1])\n    A = np.c_[(X-mu)/sd, np.ones(len(X))]\n    R = lam*np.eye(A.shape[1]); R[-1,-1]=0\n    Wr = np.linalg.solve(A.T@A+R, A.T@T)\n    # return as affine map: T = X @ M + c\n    M = Wr[:-1]/sd[:,None]; c = Wr[-1] - mu@M\n    return M, c\ndef ridge_cv(X, T, lam, folds, std=True):\n    out = np.zeros((len(X), T.shape[1]))\n    for f in range(5):\n        tr, te = folds!=f, folds==f\n        M,c = ridge_fit(X[tr], T[tr], lam, std)\n        out[te] = X[te]@M + c\n    return out\ndef report(name, lg):\n    p = lg.argmax(1)\n    print(f\"{name}: agr {np.mean(p==bp):.3f} acc {np.mean(p==y):.3f} F1 {f1macro(y,p):.3f}\")\n'''\nopen(WORK/'common.py','w').write(code)\nexec(code)\nd = WORK/'pass_3'\nlg = np.load(d/'logits.npy'); report('drop4 orig head', lg)\npooled = np.load(d/'pooled.npy'); cls = np.load(d/'cls_hidden_L5.npy')\nfor lam in [1,10,100,1000]:\n    report(f'drop4 out_proj refit pooled lam{lam}', ridge_cv(pooled, BL, lam, folds, std=False))\nfor lam in [10,100,1000]:\n    report(f'drop4 ridge cls lam{lam}', ridge_cv(cls, BL, lam, folds))\n# relative change of final cls vs base\nc6 = np.load(WORK/'pass_2/cls_hidden_L6.npy')\nprint('rel err cls', np.linalg.norm(cls-c6,axis=1).mean()/np.linalg.norm(c6,axis=1).mean())"
}
```


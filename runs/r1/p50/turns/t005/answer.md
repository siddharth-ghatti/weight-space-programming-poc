## reasoning summary

Now I'm moving to the offline analysis: checking the base model's in-sample accuracy and fitting per-layer ridge probes against two targets, one-hot labels and the base logits, with 5-fold cross-validation. I'll score each probe against the true labels and against the base model's predictions, and since ridge gives a continuous output, I'm taking the argmax for classification, with an LDA-style comparison alongside.



Next I'll run offline analysis: base in-sample accuracy, then 5-fold cross-validated ridge probes on each layer's `<s>` and mean vectors, scoring label accuracy and agreement with the base model.



## tool_use: run_python

```json
{
 "code": "import json\nd = WORK/'pass_2'\nmeta = json.load(open(d/'meta.json'))\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[x['label']] for x in P])\nassert meta['ids'] == [x['id'] for x in P] if 'ids' in meta else True\nlogits = np.load(d/'logits.npy'); bp = logits.argmax(1)\ndef f1macro(y, p, K=4):\n    f=[]\n    for k in range(K):\n        tp=((p==k)&(y==k)).sum(); fp=((p==k)&(y!=k)).sum(); fn=((p!=k)&(y==k)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\nprint('base in-sample acc', (bp==y).mean(), 'F1', f1macro(y,bp))\nnp.save(WORK/'y.npy', y); np.save(WORK/'base_logits.npy', logits)\nrng = np.random.default_rng(0); folds = rng.permutation(1000)%5\nnp.save(WORK/'folds.npy', folds)\ndef ridge_cv(X, T, lam, folds):\n    out = np.zeros((len(X), T.shape[1]))\n    for f in range(5):\n        tr, te = folds!=f, folds==f\n        mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n        Xt = (X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n        A = np.c_[Xt, np.ones(len(Xt))]\n        R = lam*np.eye(A.shape[1]); R[-1,-1]=0\n        Wr = np.linalg.solve(A.T@A+R, A.T@T[tr])\n        out[te] = np.c_[Xe, np.ones(len(Xe))]@Wr\n    return out\nY1 = np.eye(4)[y]\nfor kind in ['cls','mean']:\n    for l in range(7):\n        X = np.load(d/f'{kind}_hidden_L{l}.npy')\n        res=[]\n        for lam in [10,100,1000]:\n            pr = ridge_cv(X, Y1, lam, folds).argmax(1)\n            pl = ridge_cv(X, logits, lam, folds).argmax(1)\n            res.append(f\"lam{lam}: lab acc {np.mean(pr==y):.3f} agr {np.mean(pr==bp):.3f} | logit-fit acc {np.mean(pl==y):.3f} agr {np.mean(pl==bp):.3f}\")\n        print(kind, l, ' || '.join(res))"
}
```


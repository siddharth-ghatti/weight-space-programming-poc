## reasoning summary

Next I'll combine the results and run stratified k-fold cross-validation, testing shrinkage LDA and ridge regression on every layer for both the CLS and mean features.



## message

Both reads done. Now I'll concatenate them and evaluate each layer's CLS vs. mean features with 5-fold cross-validated closed-form classifiers (shrinkage LDA and ridge regression on one-hot targets).

## tool_use: run_python

```json
{
 "code": "import json, numpy as np\nP = practice(); lab = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab[p['label']] for p in P])\nm1 = json.load(open('pass_1/meta.json')); print({k:(v if not isinstance(v,list) else v[:2]) for k,v in m1.items()})\ndef load(name):\n    return np.concatenate([np.load(f'pass_1/{name}.npy'), np.load(f'pass_2/{name}.npy')])\nfeats = {}\nfor L in range(13):\n    feats[f'cls{L}'] = load(f'cls_hidden_L{L}'); feats[f'mean{L}'] = load(f'mean_hidden_L{L}')\nfeats['pooled'] = load('pooled')\nnp.savez_compressed('base_feats.npz', y=y, **feats)\nrng = np.random.default_rng(0); folds = np.zeros(len(y),int)\nfor c in range(8):\n    idx = np.where(y==c)[0]; rng.shuffle(idx); folds[idx] = np.arange(len(idx))%5\nnp.save('folds.npy', folds)\ndef lda_fit(X, y, alpha):\n    mu = X.mean(0); Xc = X - mu\n    means = np.stack([Xc[y==c].mean(0) for c in range(8)])\n    R = Xc - means[y]\n    S = R.T@R/len(y)\n    S = (1-alpha)*S + alpha*np.trace(S)/S.shape[0]*np.eye(S.shape[0])\n    Wt = np.linalg.solve(S, means.T).T\n    b = -0.5*np.sum(Wt*means,1) - Wt@mu\n    return Wt, b\ndef ridge_fit(X,y,lam):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\n    T=np.eye(8)[y]-1/8\n    A=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@T)\n    Wt=(A/sd[:,None]).T; b=T.mean(0)+1/8 - Wt@mu\n    return Wt,b\ndef cv(X, fit, p):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; Wt,b=fit(X[tr],y[tr],p); pred[~tr]=np.argmax(X[~tr]@Wt.T+b,1)\n    return (pred==y).mean()\nfor k in [f'{t}{L}' for L in range(13) for t in ['cls','mean']]+['pooled']:\n    X=feats[k].astype(np.float64)\n    r=[cv(X,lda_fit,a) for a in [0.1,0.3,0.6]]\n    print(k, ' '.join(f'{v:.3f}' for v in r))"
}
```


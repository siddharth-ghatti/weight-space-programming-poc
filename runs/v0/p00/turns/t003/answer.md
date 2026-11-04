## reasoning summary

I'm running 5-fold cross-validation on unigram classifiers now, comparing Naive Bayes and ridge regression on term-frequency features, with bigrams also worth checking for any gain. The features are token IDs per sentence including CLS and SEP, with TF computed as counts divided by the total token count.



I'm estimating the bag-of-wordpieces ceiling with closed-form 5-fold CV for naive Bayes and ridge, and the ridge setup matches what the network will compute.



## tool_use: run_python

```json
{
 "code": "import scipy.sparse as sp\nP = practice()\nlab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])\ntoks = tokenize([p['text'] for p in P])\nids = [t['ids'] for t in toks]\nV = 30522\nrows=[];cols=[];vals=[]\nfor i,t in enumerate(ids):\n    for j in t:\n        rows.append(i);cols.append(j);vals.append(1.0)\nX = sp.csr_matrix((vals,(rows,cols)),shape=(len(P),V))\nn = np.array([len(t) for t in ids])\nXtf = sp.diags(1.0/n) @ X\nY = np.eye(8)[y]\nrng = np.random.RandomState(0)\nperm = rng.permutation(len(P))\nfolds = np.array_split(perm,5)\ndef cv(fitfn, Xf):\n    pred = np.zeros(len(P),int)\n    for f in folds:\n        tr = np.setdiff1d(np.arange(len(P)), f)\n        S, b = fitfn(Xf[tr], y[tr], tr)\n        sc = Xf[f] @ S + b\n        pred[f] = sc.argmax(1)\n    acc = (pred==y).mean()\n    f1s=[]\n    for c in range(8):\n        tp=((pred==c)&(y==c)).sum(); fp=((pred==c)&(y!=c)).sum(); fn=((pred!=c)&(y==c)).sum()\n        f1s.append(2*tp/(2*tp+fp+fn) if tp>0 else 0)\n    return acc, np.mean(f1s), pred\ndef nb(alpha):\n    def fit(Xc, yc, tr):\n        cnt = np.zeros((V,8))\n        Xc = Xc.tocsc() if False else Xc\n        for c in range(8):\n            cnt[:,c] = np.asarray(Xc[yc==c].sum(0)).ravel()\n        lp = np.log(cnt+alpha) - np.log(cnt.sum(0)+alpha*V)\n        lp = lp - lp.mean(1,keepdims=True)\n        return lp, 0\n    return fit\nfor a in [0.05,0.1,0.3,1.0]:\n    acc,f1,_ = cv(nb(a), X)\n    print('NB counts alpha',a,acc,f1)\ndef ridge(lam):\n    def fit(Xc, yc, tr):\n        Yc = np.eye(8)[yc]\n        # dual form: S = X^T (X X^T + lam I)^-1 Y\n        Xd = Xc\n        K = (Xd @ Xd.T).toarray()\n        A = np.linalg.solve(K + lam*np.eye(K.shape[0]), Yc - Yc.mean(0))\n        S = np.asarray(Xd.T @ A)\n        b = Yc.mean(0) - np.asarray(Xd.mean(0)).ravel() @ S\n        return S, b\n    return fit\nfor lam in [1e-4,1e-3,1e-2,3e-2]:\n    acc,f1,_ = cv(ridge(lam), Xtf)\n    print('ridge TF lam',lam,acc,f1)"
}
```


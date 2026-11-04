## reasoning summary

I'm picking head 0 in layer 2 for H, since it has the smallest contribution at 1.0. Next I'm fitting D on the non-R coordinates of m2 with ridge regularization at lambda 300 on standardized values, zeroing the R columns and taking the intercept. Then I'll set the gain for R to 1 when I build the edits.

I'm setting the gain for R to 1 everywhere and the head output write gain to 1. I also want to choose the final layer 11 aggregator scale c, testing a few values in a quick numpy ridge fit, though the edited model will behave a little differently from this one.



R is [601, 222, 88, 83, 444, 506, 254, 495], so I'm sacrificing head 0 of encoder layer 2. Before building, I'll pick the mixing scale c for the [CLS] residual via a quick CV check.



## tool_use: run_python

```json
{
 "code": "lab = np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nM=np.load(WORK/'base_mean.npy'); C=np.load(WORK/'base_cls.npy')\ndef ridge_cv(X, lams):\n    res={}\n    for lam in lams:\n        pred=np.zeros(N,int)\n        for f in range(5):\n            tr=folds!=f; te=~tr\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n            Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n            W=np.linalg.solve(Xt.T@Xt+lam*np.eye(Xt.shape[1]), Xt.T@(Y[tr]-Y[tr].mean(0)))\n            pred[te]=np.argmax(Xe@W,1)\n        res[lam]=round((pred==lab).mean(),4)\n    return res\nfor c in [0.5,1,2,4,10,100]:\n    print(c, ridge_cv(c*M[11]+C[11],[30,100,300]))\n"
}
```


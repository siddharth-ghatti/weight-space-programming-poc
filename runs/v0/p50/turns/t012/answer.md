## reasoning summary

With c at 1000, the classification term barely contributes, about 0.1%, so that's fine. I'll also try a quick numpy check of the cross-validation score for normalize(M11 + C11) with c set to 1, to see whether mixing the two helps.



The norms are close, so c=1000 leaves the [CLS] residual near 0.1%; next I'll check whether a mild mix in CV scores better.



## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy')\ndef cv(X,y,lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=(X[tr].std(0)+1e-6)\n        Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n        Y=np.eye(8)[y[tr]]-1/8\n        Wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]), Xt.T@Y)\n        pred[te]=(Xe@Wt).argmax(1)\n    return (pred==y).mean()\ndef lnorm(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\nM11=np.load('feats/mean_L11.npy'); C11=np.load('feats/cls_L11.npy')\nfor c in [0.5,1,2,4,1000]:\n    print(c,[round(cv(lnorm(c*M11+C11),lab,lam),4) for lam in [50,100,200]])\nprint('concat m11,c11',[round(cv(np.concatenate([M11,C11],1),lab,lam),4) for lam in [100,200,400]])\n"
}
```


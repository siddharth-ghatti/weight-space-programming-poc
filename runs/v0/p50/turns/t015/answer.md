## message

I'll compare the edited model's final [CLS] with the expected normalize(mean11), then cross-validate on these exact features.

## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy')\nE=np.concatenate([np.load('pass_4/cls_hidden_L12.npy'),np.load('pass_5/cls_hidden_L12.npy')]); np.save('feats/edA_cls12.npy',E)\ndef lnorm(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\nM11=np.load('feats/mean_L11.npy')\npred=lnorm(M11)\nprint('max abs diff', np.abs(E-pred).max(), 'mean cos', np.mean(np.sum(E*pred,1)/np.linalg.norm(E,axis=1)/np.linalg.norm(pred,axis=1)))\ndef cv(X,y,lam):\n    pr=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=(X[tr].std(0)+1e-6)\n        Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n        Y=np.eye(8)[y[tr]]-1/8\n        Wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]), Xt.T@Y)\n        pr[te]=(Xe@Wt).argmax(1)\n    return (pr==y).mean()\nprint([ (lam,round(cv(E,lab,lam),4)) for lam in [30,60,100,150,200,300]])\n"
}
```


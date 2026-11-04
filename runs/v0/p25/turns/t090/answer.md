## reasoning summary

I'm setting up cross-validation for the kernel ridge model with an intercept, centering the targets by their weighted mean. I'll run it over the 5 folds in folds.npy and several random splits, then list the errors.



## message

Now honest CV with my own clean implementation of weighted kernel ridge, collecting the out-of-fold errors.

## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nk=np.load('kernel_v3.npz'); C=k['C']; yc=k['yc']; n=1496\nKall=gelu(C@C.T/768)\ndef fit_pred(tr, te, w, lam):\n    K=Kall[np.ix_(tr,tr)]; T=np.eye(8)[yc[tr]]; b=(w[:,None]*T).sum(0)/w.sum()\n    A=np.linalg.solve(K+lam*np.diag(1/w), T-b)\n    return Kall[np.ix_(te,tr)]@A+b\nfolds=np.load('folds.npy'); y=yc[:n]\nres={}\nfor wm in [0,0.3]:\n  for lam in [0.005,0.01,0.02]:\n    accs=[]; P=np.zeros((n,8))\n    for seed in range(4):\n        rng=np.random.default_rng(seed); f=rng.permutation(n)%5\n        for j in range(5):\n            tr=np.where(f!=j)[0]; te=np.where(f==j)[0]\n            trall=np.concatenate([tr, np.arange(n,len(yc))]) if wm>0 else tr\n            w=np.where(trall<n,1.0,wm)\n            P[te]=fit_pred(trall,te,w,lam)\n        accs.append((P.argmax(1)==y).mean())\n        if seed==0: res[(wm,lam)]=P.copy()\n    print(wm,lam,np.round(accs,4),np.mean(accs))\nnp.save('cv_oof_v3.npy',res[(0.3,0.005)])\n"
}
```


## tool_use: run_python

```json
{
 "code": "bl=np.load('base_logits.npy'); y=np.load('y.npy')\nfor f in ['lg_pass25.npy','lg_pass29.npy']:\n    lg=np.load(f); d=lg-bl\n    print(f,'mean dlogit',d.mean(0).round(3),'std',d.std(0).round(3))\n    dc=d-d.mean(1,keepdims=True); lc=lg-lg.mean(1,keepdims=True)\n    print(' corr matrix dc~lc', np.round(np.corrcoef(dc.T,lc.T)[:4,4:],2))\n    rng=np.random.default_rng(0); fo=rng.integers(0,10,1000)\n    for lam in [0,1,10,100]:\n        pred=np.zeros_like(lg)\n        for k in range(10):\n            tr=fo!=k; X=np.c_[lg[tr],np.ones(tr.sum())]\n            B=np.linalg.solve(X.T@X+lam*np.diag([1,1,1,1,0]),X.T@d[tr])\n            pred[~tr]=lg[~tr]-np.c_[lg[~tr],np.ones((~tr).sum())]@B\n        print(' lam',lam,'agr',(pred.argmax(1)==bl.argmax(1)).mean(),'rms',np.sqrt(((pred-bl)**2).mean()).round(4))\n    pred=np.zeros_like(lg)\n    for k in range(10):\n        tr=fo!=k; pred[~tr]=lg[~tr]-d[tr].mean(0)\n    print(' bias only agr',(pred.argmax(1)==bl.argmax(1)).mean(), 'rms',np.sqrt(((pred-bl)**2).mean()).round(4))\n"
}
```


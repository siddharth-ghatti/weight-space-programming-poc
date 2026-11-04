## reasoning summary

I'm going to score the kernel CV accuracy and two smoother metrics, the OOF margin and an OOF log-loss proxy, for the base model and the m1024 and m1536 variants. I'll run that on M9, M8, and M0+M9, using seeds 0 through 4 to cut down on fold noise.



## message

Now comparing base vs m=1024 (pass_19) vs m=1536 (pass_22), using kernel CV accuracy over 5 fold-seeds plus a smoother metric (mean out-of-fold margin of the true class):

## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz'); idx=np.arange(1024); yy=y[idx]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef Kf(Xs,g=1.0):\n    Z=np.concatenate([cn(X) for X in Xs],1)/np.sqrt(len(Xs)); return np.exp(g*(Z@Z.T-1))\ndef evalK(K, lam=0.005, k=10, seeds=range(5)):\n    Y=np.eye(8)[yy]; accs=[]; margs=[]\n    for s in seeds:\n        f=folds(yy,k,s); S=np.zeros((len(yy),8))\n        for i in range(k):\n            tr=f!=i; te=f==i\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n        accs.append((S.argmax(1)==yy).mean()); St=S[np.arange(len(yy)),yy]; S2=S.copy(); S2[np.arange(len(yy)),yy]=-9\n        margs.append(np.mean(np.minimum(St-S2.max(1),0.5)))\n    return '%.4f m%.4f'%(np.mean(accs),np.mean(margs))\nB=lambda l: d[f'mean_hidden_L{l}'][idx]; E1=lambda l: np.load(f'pass_19/mean_hidden_L{l}.npy'); E2=lambda l: np.load(f'pass_22/mean_hidden_L{l}.npy')\nfor nm,ls in [('9',[9]),('8',[8]),('0+9',[0,9]),('0+8',[0,8])]:\n    print(nm,'base',evalK(Kf([B(l) for l in ls])),' m1024',evalK(Kf([E1(l) for l in ls])),' m1536',evalK(Kf([E2(l) for l in ls])))"
}
```


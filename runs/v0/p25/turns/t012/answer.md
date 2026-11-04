## reasoning summary

I'm checking whether a designed mapping M, fit so that the concatenated ridge output A maps onto B, can carry the mean-pooled information without losing anything, and I'm working through what A times x comes to.

I'm noting that the combined form A cls + B mean would give the same result as A x, so the designed mapping is equivalent. Then I want to rerun the cross-validation with A and B refit inside each fold, so the labels used to build M never leak into the test split, and then evaluate LDA on the test fold.



I'm checking whether a designed mixing matrix M, fit inside each fold, lets the 768-d CLS features carry the concatenated classifier's logits; c≈2 is marginally best at about 97%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\ndef run(c, lam_cat, mode, p):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f\n        Wt,b=ridge_fit(np.hstack([c11[tr],m11[tr]]),y[tr],lam_cat)\n        A=Wt[:,:768]; B=Wt[:,768:]\n        M=c*np.eye(768)+np.linalg.pinv(A)@(B-c*A)\n        X=c11+m11@M.T\n        if mode=='direct':\n            pred[~tr]=np.argmax(X[~tr]@A.T+b,1)\n        elif mode=='lda':\n            W2,b2=lda_fit(X[tr],y[tr],p); pred[~tr]=np.argmax(X[~tr]@W2.T+b2,1)\n        else:\n            W2,b2=ridge_fit(X[tr],y[tr],p); pred[~tr]=np.argmax(X[~tr]@W2.T+b2,1)\n    return (pred==y).mean()\nfor lam in [100,300]:\n    for c in [1,2]:\n        print(lam,c,'direct',run(c,lam,'direct',0),'lda.2',run(c,lam,'lda',0.2),'ridge100',run(c,lam,'ridge',100),flush=True)"
}
```


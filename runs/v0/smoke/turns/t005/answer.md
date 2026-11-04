## reasoning summary

I'm going to fit a ridge model on the first 600 rows and check it against rows 600 to 767. I'll also try ridge regression onto one-hot encoded targets, plus LDA as another option.



## tool_use: run_python

```json
{
 "code": "X=np.load('pass_1/pooled.npy');p=practice()[:768]\ny=np.array([LABELS.index(x['label']) for x in p])\ndef fit(X,y,lam,mode):\n    Y=np.eye(8)[y]\n    mu=X.mean(0)\n    Xc=X-mu\n    if mode=='ridge':\n        Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(768),Xc.T@(Y-Y.mean(0)))\n        return Wt.T, Y.mean(0)@np.ones(8)*0+ (Y.mean(0)-Wt.T@mu)\nfor lam in [0.1,1,3,10,30,100]:\n    Wc,b=fit(X[:600],y[:600],lam,'ridge')\n    print(lam,((X[600:]@Wc.T+b).argmax(1)==y[600:]).mean(),((X[:600]@Wc.T+b).argmax(1)==y[:600]).mean())"
}
```


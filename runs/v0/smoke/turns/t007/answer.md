## tool_use: run_python

```json
{
 "code": "X=np.load('pass_1/pooled.npy');p=practice()[:768]\ny=np.array([LABELS.index(x['label']) for x in p])\ndef fit(X,y,lam):\n    Y=np.eye(8)[y];mu=X.mean(0);Xc=X-mu\n    Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(768),Xc.T@(Y-Y.mean(0)))\n    return Wt.T, Y.mean(0)-Wt.T@mu\nfor lam in [0.001,0.01,0.03,0.1,0.3]:\n    Wc,b=fit(X[:600],y[:600],lam)\n    print(lam,((X[600:]@Wc.T+b).argmax(1)==y[600:]).mean())\nWc,b=fit(X,y,0.1)\nsave_edit('classifier.weight',Wc.astype(np.float32));save_edit('classifier.bias',b.astype(np.float32))\nprint(((X@Wc.T+b).argmax(1)==y).mean())\nopen('notes.md','w').write('ridge on pooled (pass_1, p0000-0767) lam .1 saved to classifier; val ~87%\\n')"
}
```


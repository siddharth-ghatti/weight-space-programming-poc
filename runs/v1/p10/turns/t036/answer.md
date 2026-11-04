## reasoning summary

The v2 change lifted the D-only setup from 71% to 78%, but the combined score barely moved, staying around 96.4–96.6% against v1's 96.4–97.0%. Those gaps are only 1–3 items out of 472, which is noise, and both versions sit about 0.5% above the nonD-only baseline. So the carry-over gain looks small on this holdout, but it's a real if modest improvement. Now I need to look at the stage-2 problem, where the readout is trained on 1024 items with D fit in-sample and may be overfitting.

I'm weighing a fix for the stage-2 readout, which over-trusts D because it was fit in-sample. One option is to shrink D's coefficients, either by scaling the D columns or adding a heavier penalty, and another is to blend the nonD and D logits with a weight. I also want to check a random-features nonlinearity.

I'm going to check whether a random-features tanh layer with ridge does any better than the linear readout here, though at this size I doubt it. I also want to test weighting the nonD and D blocks, and maybe try an LDA-style whitening alongside ridge.



v2's carried block lifted the holdout alone (71% to 78%), but combined accuracy didn't move beyond noise (96.4–96.6% vs 96.4–97.0%). Next I'll test down-weighting the D columns in the readout, since the readout may be over-trusting them.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab=labels_all(); ytr=lab[:1024]; yte=lab[1024:]\nD0=np.load('D_dims.npy'); nonD=np.setdiff1d(np.arange(768),D0)\ndef fit_alpha(X,y,lam,alpha):\n    mu=X.mean(0); sd=X.std(0)+1e-6; w=np.ones(X.shape[1]); w[D0]=alpha\n    Xt=(X-mu)/sd*w; Y=np.eye(8)[y]\n    Wt=np.linalg.solve(Xt.T@Xt+lam*len(y)*np.eye(X.shape[1]), Xt.T@(Y-Y.mean(0)))\n    Wf=Wt*(w/sd)[:,None]; b=Y.mean(0)-mu@Wf; return Wf,b\nfor ver,(a,b_) in {'v1':('pass_3','pass_4'),'v2':('pass_5','pass_6')}.items():\n    Xtr=load('cls',12,a); Xte=load('cls',12,b_)\n    for alpha in (0.5,1,2,4,8):\n        r=[]\n        for lam in (0.05,0.1,0.2,0.4):\n            Wf,b=fit_alpha(Xtr,ytr,lam,alpha); p=(Xte@Wf+b).argmax(1); r.append('%.3f'%(p==yte).mean())\n        print(ver,'alpha',alpha,r)\n"
}
```


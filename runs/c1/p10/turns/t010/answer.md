## reasoning summary

The M0+M9 combination reaches about 97.9%, close to the original model's 97.8–98.0%, so a linear model on mean features seems to top out around 97.5–98%. The original scores 98.31% on test with 97.8% CV, so my target is at least 97.8% on test. A model with CV near 97.8% should land around 98.3% on test, and a CV of roughly 97.3% might still pass, though less reliably.

I'm weighing whether a nonlinear model on cheap features could push CV up to around 97.8%, since a CV near 97.2% would likely fall short on test. Kernel ridge, tanh random features with 768 units, and extra hidden units from the pooling FFN all look like candidates, and I want to check how far a single attention layer over embeddings can go.

I'm checking whether the pooling layer could help. The mean of layer-0 output adds little on top of M0 (96.3% combined), so early layers don't seem to contribute much linearly. The real jump comes from the deep layers, where M9 reaches 97.2% and M0+M9 reaches 97.9%, but each full layer costs about 7.09M parameters. Counting out the parameter budget for keeping layers 0 through 8 plus a pooling layer puts that option at roughly 95M, which is expensive. I'm also wondering whether the pooling step can be folded into the last real layer instead.

I'm working out the parameter budget for a deep model where the last layer's carry head reads the output of layer k-1, which lets me keep M0 and that final layer's features together. Dropping FFN width helps: at m=1024 across nine layers, the total comes to roughly 59.9M, compared with about 95M for the full-width version.

I'm weighing whether a nonlinear classifier on shallow features could reach the needed CV without paying for more layers, so I'll test RBF kernel ridge on the standardized features first.

I'm going to test kernel ridge with an RBF kernel across several gamma values on the standardized M0 features, then try 768 random tanh features with ridge on the same inputs, and finally run that same comparison on M0+M9 to see the ceiling.



Combining the embedding and layer-9 means reaches about 97.9%, close to the original model's accuracy. Before committing to an expensive deep prefix, I'll test whether a nonlinear readout of these cheap features gets near 98%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef krr_cv(X, y, gamma, lam, kind='rbf', k=5, seed=0):\n    n=len(y); S=np.zeros((n,8))\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n        Z=(X-mu)/sd; Z/=np.sqrt(Z.shape[1])\n        if kind=='rbf':\n            sq=(Z**2).sum(1)\n            K=np.exp(-gamma*(sq[:,None]+sq[None,:]-2*Z@Z.T))\n        elif kind=='poly':\n            K=(1+Z@Z.T)**gamma\n        Ktr=K[np.ix_(tr,tr)]\n        T=np.eye(8)[y[tr]]; tm=T.mean(0)\n        a=np.linalg.solve(Ktr+lam*np.eye(len(tr)), T-tm)\n        S[te]=K[np.ix_(te,tr)]@a+tm\n    return (S.argmax(1)==y).mean()\nfor L in [0,9]:\n    for g in [0.1,0.3,1.0]:\n        print(L,'rbf',g,[round(krr_cv(M[L],y,g,l),4) for l in (0.01,0.1,1)])\n    for d in [2,3]:\n        print(L,'poly',d,[round(krr_cv(M[L],y,d,l),4) for l in (0.1,1,10)])\n"
}
```


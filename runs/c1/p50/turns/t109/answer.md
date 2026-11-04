## reasoning summary

The exact KRR baselines are clearly ahead of these GELU-of-cosine approximations, and the a=3, b=-1 setting drops to about 92%, which seems off. I'm guessing the cosine values between distinct samples sit mostly near zero after centering, with same-class pairs maybe around 0.3 to 0.5, so I want to check how GELU(3cos−1) actually maps those values.

The 92% for the a=3, b=-1 setting still looks like a problem rather than a real limit, since with 1000 anchors and 1346 training points, ridge at a small λ should fit fairly well. I'm wondering whether the Φ normalization is shrinking each feature so much that λ=0.001 dominates, which would explain why accuracy stays flat across λ.

I'm checking the Φ normalization against the λ scale. The trace of FᵀF is about 1346 spread over 1000 dimensions, so the tail eigenvalues are small, but λ=0.001 still looks like the right order. Then I'm asking why a=3, b=-1 does so much worse: GELU(3cos−1) maps the self-match to about 1.95 and other pairs to roughly −0.16 to 0.35, which should be workable. I want to test this directly by computing exact KRR with the same kernel.

I'm going to test whether the empirical kernel map with ridge matches exact KRR using the same GELU kernel, and also check an exp-based feature version. If the exp version is also poor, the bug is probably in my anchor CV code rather than the kernel choice.



The anchor-feature results sit well below exact KRR (98.2–98.5%), and the flat 92% for a=3, b=−1 looks like a bug. I'll check the procedure with exp(S−1) anchor features and compare against the dual form.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\nZ9=cn(d['mean_hidden_L9'])\nS=Z9@Z9.T\nprint('cos stats offdiag', np.percentile(S[~np.eye(N,dtype=bool)],[1,25,50,75,99]))\nf=folds(y,10,0); tr=np.where(f!=0)[0]; te=np.where(f==0)[0]\nfor nm,fn in [('exp',lambda s: np.exp(s-1)),('gelu1',lambda s: gelu(s)),('gelu3-1',lambda s: gelu(3*s-1))]:\n    Phi=fn(S[:, tr])  # all train as anchors\n    for lam in [1e-3,1e-2,1e-1,1]:\n        Wt=np.linalg.solve(Phi[tr].T@Phi[tr]+lam*np.eye(len(tr)), Phi[tr].T@(Y[tr]-Y[tr].mean(0)))\n        acc_p=((Phi[te]@Wt).argmax(1)==y[te]).mean()\n        K=fn(S); A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)), Y[tr]-Y[tr].mean(0)); acc_d=((K[np.ix_(te,tr)]@A).argmax(1)==y[te]).mean()\n        print(nm,lam,'primal-anchor',round(acc_p,4),'dual KRR',round(acc_d,4))"
}
```


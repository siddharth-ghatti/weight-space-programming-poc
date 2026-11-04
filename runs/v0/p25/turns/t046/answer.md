## reasoning summary

I'm writing a kernel cross-validation function: standardize the features using statistics from all the data, which is fine since they don't use labels, then compute the scaled kernel and apply the GELU nonlinearity. For prediction I'd use the dual ridge solution, with test points mapped through the same kernel and the training mean subtracted from the targets.

I'm working out how v2 handled the ridge step, guessing that the features are the Gelu of the scaled kernel over the training centers, and that the weights come from a regularized solve against the centered targets. In each CV fold the centers come only from the training points. Next I'll implement this and compare several feature sets across 5-fold CV with a few random splits.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nimport time\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\ndef kcv(X, y, beta=1.0, lam=0.01, nfold=5, seed=0, center_frac=None):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; d=X.shape[1]\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%nfold\n    pred=np.zeros(len(y),int)\n    for k in range(nfold):\n        tr=fo!=k; te=~tr\n        P=gelu(beta*Z[tr]@Z[tr].T/d); T=np.eye(8)[y[tr]]\n        Pm=P.mean(0); Tm=T.mean(0)\n        Pc=P-Pm; A=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n        Pt=gelu(beta*Z[te]@Z[tr].T/d)\n        pred[te]=np.argmax((Pt-Pm)@A+Tm,1)\n    return (pred==y).mean()\nt=time.time()\nX=LN(f['cls11']+2*f['mean11'])\nprint([round(kcv(X,y,seed=s),4) for s in range(3)], time.time()-t)\n"
}
```


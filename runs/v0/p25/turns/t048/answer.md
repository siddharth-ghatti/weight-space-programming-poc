## reasoning summary

The numbers are all within about 0.978 to 0.984, and with 1496 samples the standard error is around 0.0035, so these gaps are likely noise. Adding mean0 nudges it to 0.984, and multi-layer means reach 0.983. I want to check beta and lambda tuning on the v2 features and the m8 through m11 variants, and also test pooling across all layers. A realistic gain might be around 0.3%, so I'm weighing whether that justifies the added complexity and risk.

I'm going to try a more systematic approach, starting with hyperparameters, since each new edit carries some risk and I have plenty of budget to spend on it.



## message

Differences between feature sets are within noise (SE ≈ 0.0035). Next I'll tune beta/lambda for the current v2 features and a couple of candidate sets.

## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\ndef kcv(X, y, beta=1.0, lam=0.01, nfold=5, seed=0):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; d=X.shape[1]\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%nfold\n    pred=np.zeros(len(y),int)\n    for k in range(nfold):\n        tr=fo!=k; te=~tr\n        P=gelu(beta*Z[tr]@Z[tr].T/d); T=np.eye(8)[y[tr]]\n        Pm=P.mean(0); Tm=T.mean(0)\n        Pc=P-Pm; A=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n        Pt=gelu(beta*Z[te]@Z[tr].T/d)\n        pred[te]=np.argmax((Pt-Pm)@A+Tm,1)\n    return (pred==y).mean()\ndef ev(X, **kw): return np.mean([kcv(X,y,seed=s,**kw) for s in range(5)])\nX=LN(f['cls11']+2*f['mean11'])\nfor beta in [0.5,1,2,4]:\n    print(beta, [round(ev(X,beta=beta,lam=l),4) for l in [0.001,0.01,0.1,1]])\n"
}
```


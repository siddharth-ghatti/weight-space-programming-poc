## reasoning summary

Next I'll try the feature sets: concatenation, with each block standardized, where the kernel comes out as Z·Zᵀ/d. Then I'll run a scan over seeds 0 through 4, along with several betas and lambda values.



## message

That matches the notes (~0.98). Now I'll scan feature sets (kernel on combinations of layers) to see if anything clearly beats it.

## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\ndef kcv(X, y, beta=1.0, lam=0.01, nfold=5, seed=0):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; d=X.shape[1]\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%nfold\n    pred=np.zeros(len(y),int)\n    for k in range(nfold):\n        tr=fo!=k; te=~tr\n        P=gelu(beta*Z[tr]@Z[tr].T/d); T=np.eye(8)[y[tr]]\n        Pm=P.mean(0); Tm=T.mean(0)\n        Pc=P-Pm; A=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n        Pt=gelu(beta*Z[te]@Z[tr].T/d)\n        pred[te]=np.argmax((Pt-Pm)@A+Tm,1)\n    return (pred==y).mean()\ndef ev(X, **kw): return np.mean([kcv(X,y,seed=s,**kw) for s in range(5)])\nsets={\n 'v2 LN(c11+2m11)': LN(f['cls11']+2*f['mean11']),\n 'c11|m11': np.hstack([f['cls11'],f['mean11']]),\n 'm11': f['mean11'], 'm10':f['mean10'], 'm12':f['mean12'], 'c12':f['cls12'],\n 'm10|m11': np.hstack([f['mean10'],f['mean11']]),\n 'c11|m11|m0': np.hstack([f['cls11'],f['mean11'],f['mean0']]),\n 'c11|m11|m6': np.hstack([f['cls11'],f['mean11'],f['mean6']]),\n 'm8|m9|m10|m11':np.hstack([f['mean8'],f['mean9'],f['mean10'],f['mean11']]),\n 'c10|m10': np.hstack([f['cls10'],f['mean10']]),\n}\nfor k,X in sets.items(): print(k, round(ev(X),4))\n"
}
```


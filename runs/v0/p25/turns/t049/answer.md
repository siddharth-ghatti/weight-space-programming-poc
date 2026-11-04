## reasoning summary

The current setting, beta=1 with lam=0.01, already looks like the best option, so I'll keep it. Beta=4 collapsing is odd, and I suspect numerical trouble from GELU behaving like a sharp ReLU at large beta. I'm also considering an ensemble of the kernel model with a linear one, such as LDA on the cls and mean features.

I'm weighing a bias shift and cosine-normalized centers as ways to get something RBF-like out of gelu(beta*z.zi + b), so I'll try both.



The current setting (beta=1, lambda=0.01) is already at the CV optimum, so I'm trying kernel variants a GELU neuron can implement: a global bias shift and unit-length centers.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\ndef kcv(X, y, beta=1.0, lam=0.01, nfold=5, seed=0, shift=0.0, norm=False):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; d=X.shape[1]\n    C=Z/np.linalg.norm(Z,axis=1,keepdims=True)*np.sqrt(d) if norm else Z\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%nfold\n    pred=np.zeros(len(y),int)\n    for k in range(nfold):\n        tr=fo!=k; te=~tr\n        P=gelu(beta*Z[tr]@C[tr].T/d+shift); T=np.eye(8)[y[tr]]\n        Pm=P.mean(0); Tm=T.mean(0)\n        Pc=P-Pm; A=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n        Pt=gelu(beta*Z[te]@C[tr].T/d+shift)\n        pred[te]=np.argmax((Pt-Pm)@A+Tm,1)\n    return (pred==y).mean()\ndef ev(X, **kw): return np.mean([kcv(X,y,seed=s,**kw) for s in range(5)])\nX=LN(f['cls11']+2*f['mean11'])\nfor sh in [-1,-0.5,0.5,1]:\n    print('shift',sh,[round(ev(X,shift=sh,lam=l),4) for l in [0.003,0.01,0.03]])\nfor beta in [0.5,1,2]:\n    print('norm beta',beta,[round(ev(X,norm=True,beta=beta,lam=l),4) for l in [0.003,0.01,0.03]])\n"
}
```


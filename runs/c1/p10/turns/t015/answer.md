## reasoning summary

I'm going to push the lambda values lower, down to 1e-6 and 1e-7, since smaller values seem to work better here. The tanh and exp versions gave identical results, which suggests the approximation holds. I also notice this primal ridge over 768 kernel features is essentially a subset-of-regressors approach, and I'm scaling lambda by trace over n.



## message

Lower penalties are clearly better, and tanh with a negative bias matches exp exactly, so the approximation works. I'll extend the penalty grid downwards.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\nexec(open(WORK/'lib.py').read())\ndef center_cv(X, y, nunits, gamma, lams, k=5, seed=0, rs=0):\n    n=len(y); S={l:np.zeros((n,8)) for l in lams}\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n        rng=np.random.default_rng(rs)\n        ci = rng.choice(tr,nunits,replace=False) if nunits<len(tr) else tr\n        Hf=np.exp(2*gamma*Z@Z[ci].T - 2*gamma)\n        Ht=Hf[tr]; T=np.eye(8)[y[tr]]-1/8\n        G=Ht.T@Ht; sc=np.trace(G)/G.shape[0]\n        w,V=np.linalg.eigh(G); R=V.T@(Ht.T@T)\n        for l in lams:\n            Wt=V@(R/(w+l*sc)[:,None]); S[l][te]=(Hf@Wt)[te]\n    return {l:round((S[l].argmax(1)==y).mean(),4) for l in lams}\nlams=(1e-7,1e-6,1e-5,3e-5)\nfor L in [0,1,9]:\n    X=LNr(M[L])\n    for g in (0.2,0.4,0.8):\n        print(L,g,'768',center_cv(X,y,768,g,lams),'all',center_cv(X,y,5000,g,lams))\n"
}
```


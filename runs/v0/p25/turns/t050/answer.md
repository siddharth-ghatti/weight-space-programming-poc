## reasoning summary

The model's already close to the best it can get, though a shift of -1 dropping to 0.91 looks off and may point to ill-conditioning in the solve with a small lambda on the gelu features. I want to check which items keep getting misclassified across folds, and whether the errors cluster between conditions_duties and claims_procedure in a way that targeted features could address.



Small kernel tweaks didn't help, so v2's settings look near-optimal here. I'm checking whether the consistently misclassified sentences in CV share a pattern or just have ambiguous labels.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\ndef kcvp(X, y, beta=1.0, lam=0.01, nfold=5, seed=0):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; d=X.shape[1]\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%nfold\n    pred=np.zeros(len(y),int)\n    for k in range(nfold):\n        tr=fo!=k; te=~tr\n        P=gelu(beta*Z[tr]@Z[tr].T/d); T=np.eye(8)[y[tr]]\n        Pm=P.mean(0); Tm=T.mean(0)\n        Pc=P-Pm; A=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n        Pt=gelu(beta*Z[te]@Z[tr].T/d)\n        pred[te]=np.argmax((Pt-Pm)@A+Tm,1)\n    return pred\nX=LN(f['cls11']+2*f['mean11'])\nP=np.stack([kcvp(X,y,seed=s) for s in range(5)])\nerr=(P!=y).sum(0)\npr=practice(); \nfrom collections import Counter\nprint('items wrong in >=3 of 5 splits:', (err>=3).sum(), ' any:', (err>0).sum())\nC=Counter()\nfor i in np.where(err>=3)[0]:\n    pm=Counter(P[:,i]).most_common(1)[0][0]\n    C[(LABELS[y[i]],LABELS[pm])]+=1\n    print(f\"{LABELS[y[i]][:12]:12s} -> {LABELS[pm][:12]:12s} | {pr[i]['text'][:150]}\")\nprint(C.most_common())\n"
}
```


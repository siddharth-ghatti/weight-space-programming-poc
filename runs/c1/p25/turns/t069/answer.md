## reasoning summary

I've pinned down the exact relation: the centered reserved-point value is a linear combination of the centered s2 and s6 terms, with coefficients about 3.87 and 3.05, residual near 1e-5. Next I'm setting up cross-validation on the model I can actually implement: z-scoring the non-reserved inputs per dimension using stats from all 1496 points, then building the kernel with GELU.

I'm now defining the GELU kernel with the coefficients a, b, c and the scaling that compares against the z(M9) version, plus the linear κ-weighted terms for M2 and M6 with reserved dimensions handled.



The carry model fits exactly, with residual around 1e-5, giving κ2 ≈ 3.87 and κ6 ≈ 3.05. Next I'm comparing the CV of the implementable model against the idealized M9 version.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom cvlib import folds\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy'); keep=np.setdiff1d(np.arange(768),R)\nU=np.load('U_k10.npy'); kk=np.load('kappa_k10.npz'); K2,K6=kk['K2'],kk['K6']\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, lams, seeds=range(5)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=K[np.ix_(te,tr)]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\nZu=z(U[:,keep]); Tu=Zu@Zu.T/760\nZm=z(d['mean_hidden_L9'][:,keep]); Tm=Zm@Zm.T/760\nZ2=z(d['mean_hidden_L2'][:,keep]); Z6=z(d['mean_hidden_L6'][:,keep])\nL2=Z2@Z2.T/760; L6=Z6@Z6.T/760\nk2=K2/K2.mean(); k6=K6/K6.mean()\nL2k=np.outer(k2,k2)*L2; L6k=np.outer(k6,k6)*L6\nlams=(1e-2,3e-2,1e-1)\nprint('M9 gelu + lin', kcv(2*gelu(Tm)+L2+L6,lams))\nprint('U gelu + lin', kcv(2*gelu(Tu)+L2+L6,lams))\nprint('U gelu + lin kappa', kcv(2*gelu(Tu)+L2k+L6k,lams))\nfor a in (0.7,1.3):\n    print('U gelu a',a,'+lin kappa', kcv(2*gelu(a*Tu)/a+L2k+L6k,lams))\nprint('U gelu + 0.5lin kappa', kcv(2*gelu(Tu)+0.5*(L2k+L6k),lams))\nprint('U gelu + 2lin kappa', kcv(2*gelu(Tu)+2*(L2k+L6k),lams))"
}
```


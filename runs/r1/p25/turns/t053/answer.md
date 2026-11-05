## reasoning summary

The data-driven magnitude-times-variance selection comes out ahead, while the OBS version with Hessian regularization does worse, probably from overfitting the Hessian with only 3000 samples. Next I want to see how the error changes as the amount of training data changes, so I'm considering training on two folds.

I'm checking whether the amount of training data matters, so I'm comparing a two-fold training split against a three-fold one, since a large gap would suggest more data is worth the extra compute.



Simple data-based selection beats both OBS and Gaussian-moment importance. I'm next testing whether more probe data helps by comparing 16 and 24 paragraphs on held-out error, since a large gap would justify extra token passes for layers 0–2.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy')\ndef run4(l, m, ntrain_folds, lam=200., wcls=20.0, nf=8):\n    fold=grp%nf\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2\n    num=0; den=0; num0=0\n    for f in range(nf):\n        te=fold==f; trf=[(f+1+j)%nf for j in range(ntrain_folds)]; tr=np.isin(fold,trf)\n        w=np.where(tk[tr]==0,wcls,1.0); sw=np.sqrt(w)[:,None]\n        Gt=G[tr]; mu=(w[:,None]*Gt).sum(0)/w.sum()\n        Gc=(Gt-mu)*sw; un=(U**2).sum(0)\n        sc=un*(Gc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n        A=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n        D=np.linalg.solve(A+lam*np.eye(m),B).T\n        Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n        Fe0=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)\n        e=((Fe-F[te])**2).sum(1); e0=((Fe0-F[te])**2).sum(1)\n        Fm=F[tr][tk[tr]==2].mean(0)\n        num+=e[tk[te]==2].sum(); num0+=e0[tk[te]==2].sum(); den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n    return round(num0/den,4), round(num/den,4)\nfor l in (0,1,2):\n    print(l, [ (nt, run4(l,2048,nt)) for nt in (2,4,7)])\n"
}
```


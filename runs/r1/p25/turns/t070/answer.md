## reasoning summary

I'm checking that the last token is </s>, assuming the harness's truncation adds it at the end. Then I want a quick layer-0 test first: cross-validate with old plus new data, training on new plus three-quarters of old and testing on held-out old folds, to see whether the extra data helps before spending passes on L1 and L2.



## message

Before spending passes on layers 1 and 2, I'll confirm on layer 0 that the extra long-paragraph tokens actually reduce held-out error on the original probe paragraphs, at m=1536 and 2048.

## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy'); fold=grp%4\na2=np.load('true2_a_L0.npy').astype(np.float64); tk2=np.load('tok2_kind.npy')\nl=0; pre=f'roberta.encoder.layer.{l}.'\na=np.load(f'true_a_L{l}.npy').astype(np.float64)\nW1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\nU=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\nG=gelu(a@W1.T+b1); F=G@U.T+b2; G2=gelu(a2@W1.T+b1)\ndef cvx(m, lam, use_new, wcls=20., weos=5.):\n    num=0; den=0\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        Gt=G[tr]; kt=tk[tr]\n        if use_new: Gt=np.vstack([Gt,G2]); kt=np.concatenate([kt,tk2])\n        w=np.where(kt==0,wcls,np.where(kt==1,weos,1.0)); sw=np.sqrt(w)[:,None]\n        mu=(w[:,None]*Gt).sum(0)/w.sum(); Gc=(Gt-mu)*sw\n        sc=(U**2).sum(0)*(Gc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n        A=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n        D=np.linalg.solve(A+lam*np.eye(m),B).T\n        Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n        e=((Fe-F[te])**2).sum(1)\n        Fm=F[tr][tk[tr]==2].mean(0)\n        num+=e[tk[te]==2].sum(); den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n    return round(num/den,4)\nfor m in (1536,2048):\n    print(m,'old only lam100',cvx(m,100,False),' +new:',[(lam,cvx(m,lam,True)) for lam in (30,100,300)])\n"
}
```


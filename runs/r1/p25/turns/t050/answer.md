## reasoning summary

I'm setting up the layer-wise least-squares experiment: for a given layer, I take the activations, pass them through the first linear map and GELU, and form the projected output. Units get ranked by a data-based importance score, and for the pruned units I'm working out a bias correction so the remaining output still matches.

I'm working out the closed-form least-squares refit for the kept units, solving for the update that compensates the pruned units' contribution with a ridge penalty. I also want to upweight the CLS rows, since there are only 32 of them, and check the relative error separately for each token kind on held-out data. The matrices are small enough that the solve is cheap.



Confirmed the token kinds: <s> is kind 0, </s> is kind 1, and ordinary tokens are kind 2. Next I'll run the held-out ridge refit with <s> rows upweighted, evaluated on paragraph folds.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy')\nfold=grp%4\ndef run(l, m, lams, wcls=20.0, sel='data'):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2\n    res={lam:{0:[],1:[],2:[]} for lam in lams}; den={0:[],1:[],2:[]}\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        w=np.where(tk[tr]==0,wcls,1.0)\n        Gt=G[tr]; mu=(w[:,None]*Gt).sum(0)/w.sum()\n        if sel=='data':\n            sc=(U**2).sum(0)*((w[:,None]*(Gt-mu)**2).sum(0)/w.sum())\n        keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n        Gc=(Gt[:,keep]-mu[keep])*np.sqrt(w)[:,None]\n        R=((Gt[:,pr]-mu[pr])@U[:,pr].T)*np.sqrt(w)[:,None]\n        A=Gc.T@Gc; B=Gc.T@R\n        for lam in lams:\n            if lam is None: D=np.zeros((768,m))\n            else: D=np.linalg.solve(A+lam*np.eye(m),B).T\n            Un=U[:,keep]+D; b2n=b2+U[:,pr]@mu[pr]-D@mu[keep]\n            Fe=G[te][:,keep]@Un.T+b2n\n            e=((Fe-F[te])**2).sum(1)\n            for k in range(3): res[lam][k].append(e[tk[te]==k])\n        Fm=F[tr][tk[tr]==2].mean(0)\n        for k in range(3): den[k].append(((F[te]-Fm)**2).sum(1)[tk[te]==k])\n    out=[]\n    for lam in lams:\n        out.append(f'lam={lam}: '+' '.join(f'k{k} {np.concatenate(res[lam][k]).sum()/np.concatenate(den[k]).sum():.4f}' for k in range(3)))\n    return out\nfor l in range(5):\n    for m in (1536,2048,2560):\n        print(l,m,run(l,m,[None,1.0,10.0,100.0]))\n"
}
```


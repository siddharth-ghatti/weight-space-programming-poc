## reasoning summary

I've settled that linear units hurt when combined with least squares, since the LS fit on the kept units already captures the linear part, so I'll use r=0 with λ around 100 to 300. For selection, I want to compare ranking by the Gaussian I criterion against ranking by the data, and also test a leverage-style score that discounts units whose contribution is already predicted by the others, perhaps with a cheap OBS-style greedy removal.

I'm weighing OBS-style greedy pruning as a leverage-type selection: each unit's removal cost is its output-weight norm divided by the diagonal of the inverse Gram matrix, with the remaining weights updated after each removal. I'm checking whether that greedy, recalculated process still counts as closed-form or drifts toward iterative fitting, since a single selection followed by one LS solve clearly fits the constraint.

I'm now weighing whether one-shot OBS scores, computed once from the inverse of the regularized Gram matrix, beat plain magnitude scores. With the matrix rank-deficient, units explained by the others get large diagonal inverse entries and so low scores, which is the pruning signal I want. I'm also concerned about overfitting, since there are only about 3000 training tokens, and I'm wondering whether the limit of 32 ids per probe pass can be worked around for the early layers.

Each probe pass only yields one layer's FFN input, since the hidden state holds one layer's output at a time and the 768-dim state can't carry both h_{l+1} and a_l. So getting several layers at once from a single pass isn't possible this way.

With only 32 ids per pass, one layer per pass, I can't pull several layers' FFN inputs at once. I'm thinking about spending three of the 14 remaining passes on extra tokens for layers 0 through 2, then checking with CV whether 16 versus 24 ids per layer changes the error. First, though, I'll test the OBS-style score selection.



Adding linear units on the LS refit hurt, so I'm dropping that. Next I'll test OBS-based unit selection against magnitude×variance.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy')\nfold=grp%4\ndef run3(l, m, sels, lam=200., hl=(10.,100.,1000.), wcls=20.0):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2\n    gI=np.load(f'timp_L{l}.npz')['I']\n    num={}; den=0\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        w=np.where(tk[tr]==0,wcls,1.0); sw=np.sqrt(w)[:,None]\n        Gt=G[tr]; mu=(w[:,None]*Gt).sum(0)/w.sum()\n        Gc=(Gt-mu)*sw\n        H=Gc.T@Gc\n        un=(U**2).sum(0)\n        scores={'data':un*np.diag(H),'gauss':gI}\n        for h in hl:\n            Hi=np.linalg.inv(H+h*np.eye(3072))\n            scores[f'obs{h:g}']=un/np.diag(Hi)\n        Fm=F[tr][tk[tr]==2].mean(0)\n        den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n        for name,sc in scores.items():\n            keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n            A=H[np.ix_(keep,keep)]; B=H[np.ix_(keep,pr)]@U[:,pr].T\n            D=np.linalg.solve(A+lam*np.eye(m),B).T\n            Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n            e=((Fe-F[te])**2).sum(1)\n            num[name]=num.get(name,0)+e[tk[te]==2].sum()\n    return {k:round(v/den,4) for k,v in num.items()}\nfor l in (0,1,2):\n    print(l, run3(l,2048,None))\n"
}
```


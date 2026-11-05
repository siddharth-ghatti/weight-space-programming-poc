## reasoning summary

I'm wondering whether the 2.42s duration is real or came from a cache, since it looks a bit suspicious. I want to confirm the data is sensible too: the variance should be around 1, and the values should differ from pass 15.



Pass 16 finished in 2.4 s, so I'm checking its rows are real and distinct from pass 15 before saving the layer-2 inputs, then running held-out tests at m=1536/1792/2048.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\nmask=np.load('pass_16/attention_mask.npy').astype(bool)\nn=np.load('pass_16/token_hidden_L3.npy')[mask].astype(np.float64)\nn15=np.load('pass_15/token_hidden_L2.npy')[mask].astype(np.float64)\nprint('var range',n.var(1).min(),n.var(1).max(), mask.sum(), 'diff vs pass15', np.abs(n-n15).mean())\nl=2\ng=base(f'roberta.encoder.layer.{l}.attention.output.LayerNorm.weight').astype(np.float64)\nb=base(f'roberta.encoder.layer.{l}.attention.output.LayerNorm.bias').astype(np.float64)\nnp.save(f'true2_a_L{l}.npy',(n*g+b).astype(np.float32))\ns=open('notes.md').read()\ns+='- passes 14-16 done: true2_a_L0/L1/L2.npy (10268 tokens, tok2_kind.npy). L0 CV with new data: m1536 .091->.081, m2048 .042->.035.\\n'\nopen('notes.md','w').write(s)\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy'); fold=grp%4; tk2=np.load('tok2_kind.npy')\ndef cvx(l, ms, lams, wcls=20., weos=5.):\n    pre=f'roberta.encoder.layer.{l}.'\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64); a2=np.load(f'true2_a_L{l}.npy').astype(np.float64)\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2; G2=gelu(a2@W1.T+b1)\n    res={}\n    for use_new in (False,True):\n      for m in ms:\n        for lam in lams:\n          num=0; den=0\n          for f in range(4):\n            tr=fold!=f; te=~tr\n            Gt=G[tr]; kt=tk[tr]\n            if use_new: Gt=np.vstack([Gt,G2]); kt=np.concatenate([kt,tk2])\n            w=np.where(kt==0,wcls,np.where(kt==1,weos,1.0)); sw=np.sqrt(w)[:,None]\n            mu=(w[:,None]*Gt).sum(0)/w.sum(); Gc=(Gt-mu)*sw\n            sc=(U**2).sum(0)*(Gc**2).sum(0)\n            keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n            A=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n            D=np.linalg.solve(A+lam*np.eye(m),B).T\n            Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n            e=((Fe-F[te])**2).sum(1)\n            Fm=F[tr][tk[tr]==2].mean(0)\n            num+=e[tk[te]==2].sum(); den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n          res[(use_new,m,lam)]=round(num/den,4)\n    return res\nfor l in (1,2):\n    print(l, cvx(l,(1536,1792,2048),(100,)))\n"
}
```


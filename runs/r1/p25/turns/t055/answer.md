## reasoning summary

Layer 5 at m=2048 shows 0.5% error, which is good, and at 1536 it's 1.4%. Next I'm setting up an m=2048 edit across all six layers, with token-level settings on layers 0 through 4 and a CLS-only setting on layer 5, then running a read that pulls the CLS hidden states, mean hidden states, and logits.

I'm also planning to save the builder script as lsbuild.py so I can reuse it, and the read pass will cover all 1000 for the agreement check and head refit, which makes one pass in total.



Layer 5 compresses well, at 0.5% error for m=2048 and 1.4% for m=1536. Next I'll write a reusable LS builder and save the m=2048 edit for all six layers.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ndef layer_data(l, base):\n    pre=f'roberta.encoder.layer.{l}.'\n    a=np.load('true_a5_cls.npy' if l==5 else f'true_a_L{l}.npy').astype(np.float64)\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    return a,W1,b1,U,b2\ndef build_ls(l, m, base, lam=None, wcls=20.0, weos=5.0, extra=None):\n    a,W1,b1,U,b2=layer_data(l,base)\n    if extra is not None:\n        a=np.vstack([a,extra[0]])\n    G=gelu(a@W1.T+b1)\n    if l==5:\n        w=np.ones(len(a)); lam=10.0 if lam is None else lam\n    else:\n        tk=np.load('tok_kind.npy')\n        if extra is not None: tk=np.concatenate([tk,extra[1]])\n        w=np.where(tk==0,wcls,np.where(tk==1,weos,1.0)); lam=200.0 if lam is None else lam\n    sw=np.sqrt(w)[:,None]\n    mu=(w[:,None]*G).sum(0)/w.sum(); Gc=(G-mu)*sw\n    sc=(U**2).sum(0)*(Gc**2).sum(0)\n    keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n    A=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n    D=np.linalg.solve(A+lam*np.eye(m),B).T\n    Un=U.copy(); Un[:,keep]+=D; Un[:,pr]=0\n    b2n=b2+U[:,pr]@mu[pr]-D@mu[keep]\n    return keep, Un, b2n\ndef apply_ls(m, base, W, save_edit, layers=range(6), **kw):\n    keep={}\n    for l in layers:\n        pre=f'roberta.encoder.layer.{l}.'\n        k,Un,b2n=build_ls(l,m,base,**kw)\n        save_edit(pre+'output.dense.weight',Un.astype(np.float32))\n        save_edit(pre+'output.dense.bias',b2n.astype(np.float32))\n        save_edit(pre+'intermediate.dense.weight',base(pre+'intermediate.dense.weight'))\n        save_edit(pre+'intermediate.dense.bias',base(pre+'intermediate.dense.bias'))\n        keep[l]=k.tolist()\n    return keep\n'''\nopen('lsbuild.py','w').write(code)\nexec(code)\nkeep=apply_ls(2048, base, W, save_edit)\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep=keep))\nprint(edits()[:30])\n"
}
```


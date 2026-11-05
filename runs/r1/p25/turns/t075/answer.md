## reasoning summary

I've got a mixed (1,1) setup at λ=30 scoring 0.9563 on the edited inputs and 0.9864 on the base ones, so it holds up well, and I'll use it for layer 5. The main limit is upstream error: the final h6 R² sits around 0.956 for m=1536 on the old data, and new data for L0–L2 might lift that to about 0.962. Next I'm turning to the head and building candidate C at m=1536 to evaluate.

Next I'm weighing whether to test m=1792 as well, since m=1536 only wins if agreement clears about 0.98 with a low-margin score near 0.93, with m=1792 as the fallback and m=2048 after that. I'm building candidate C with the extra data for layers 0–2 so I can run the logits and CLS hidden pass and see how it does.



Layer 5 mixed-fit gave the best held-out edited-input result (0.9563) while staying accurate on base inputs (0.986), so I'm now assembling the full m=1536 candidate with per-layer data and λ settings.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nexec(open('lsbuild.py').read())\ncode2 = r'''\ndef build_l5_mixed(m, base, ae_file, lam=30.0, wb=1.0, we=1.0):\n    def LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\n    ab,W1,b1,U,b2=layer_data(5,base)\n    ae=np.load(ae_file).astype(np.float64)\n    Gb=gelu(ab@W1.T+b1); Ge=gelu(ae@W1.T+b1); xb=ab+Gb@U.T+b2\n    X=np.vstack([Gb,Ge]); T=np.vstack([xb-ab, xb-ae]); w=np.concatenate([np.full(len(ab),wb),np.full(len(ae),we)])\n    sw=np.sqrt(w)[:,None]; mu=(w[:,None]*X).sum(0)/w.sum(); tm=(w[:,None]*T).sum(0)/w.sum()\n    Xc=(X-mu)*sw; sc=(U**2).sum(0)*(Xc**2).sum(0)\n    keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n    Xk=Xc[:,keep]; R=(T-tm)*sw-Xk@U[:,keep].T\n    D=np.linalg.solve(Xk.T@Xk+lam*np.eye(m),Xk.T@R).T\n    Un=np.zeros_like(U); Un[:,keep]=U[:,keep]+D\n    b2n=tm-Un[:,keep]@mu[keep]\n    return keep,Un,b2n\ndef build_candidate(m, base, save_edit, drop_edit, edits, ae_file=None, lam=100.0, lam5=30.0):\n    for e in edits(): drop_edit(e)\n    keep={}\n    for l in range(5):\n        extra=None\n        if l<=2:\n            extra=(np.load(f'true2_a_L{l}.npy').astype(np.float64), np.load('tok2_kind.npy'))\n        k,Un,b2n=build_ls(l,m,base,lam=lam,extra=extra)\n        pre=f'roberta.encoder.layer.{l}.'\n        save_edit(pre+'output.dense.weight',Un.astype(np.float32)); save_edit(pre+'output.dense.bias',b2n.astype(np.float32))\n        keep[l]=k.tolist()\n    if ae_file is None:\n        k,Un,b2n=build_ls(5,m,base)\n    else:\n        k,Un,b2n=build_l5_mixed(m,base,ae_file,lam=lam5)\n    pre='roberta.encoder.layer.5.'\n    save_edit(pre+'output.dense.weight',Un.astype(np.float32)); save_edit(pre+'output.dense.bias',b2n.astype(np.float32))\n    keep[5]=k.tolist()\n    return keep\n'''\nopen('lsbuild.py','a').write(code2)\nexec(code2)\nkeep=build_candidate(1536, base, save_edit, drop_edit, edits, ae_file='a5e_1536.npy')\njson.dump({str(k):v for k,v in keep.items()},open('keep_C1536.json','w'))\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep=keep))\nprint(edits())\n# sanity: LN params of layer5 are base\nprint(np.allclose(W('roberta.encoder.layer.5.output.LayerNorm.weight'),base('roberta.encoder.layer.5.output.LayerNorm.weight')))\n"
}
```


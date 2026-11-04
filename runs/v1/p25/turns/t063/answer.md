## reasoning summary

I'm checking the scale of the layer norm output: the token RMS sits around 0.7, which fits a unit-RMS normalized vector scaled by gamma near 0.85. Next I'm working out how the pre-norm residual sum u = h + a relates to the sigma computed on u, since gamma over sigma maps u's scale onto h's.

Now I'm setting the stash scale: I want γ_S near σ so the ratio stays near 1, and σ looks like roughly 0.8 from RMS(h) and RMS(a). I'll try the median γ of each LayerNorm and measure. Next is writing a stash.py that builds the edits for layers 0 through 8 from the base weights, with the ablation means taken from the original model.

Still writing the stash edits for layer 0, zeroing out the query rows and setting the value projection for the J dims...

Now I'm working out the stash edits for layers 1 through 8: each layer's Q, K, and V reads from the stashed dimensions get replaced by constant means, with the FFN read handled the same way. The S columns in W1 and the S rows in W2 are zeroed out, and the LayerNorm gains for those dimensions are set to the median values with zero bias. In layer 0 the value rows for the J dimensions are set to A0, so the uniform average over tokens gives A0·M0 + b0, which should match the reference mean.

Still working through the stash edits: the attention output and FFN reads for those dimensions are set to constant means, and the layer-9 readout uses the same stashed values until I refit it.

I'm weighing whether to run a 256-sample diagnostic pass first, since a redo seems plausible, against reading all 1496 samples directly with cls hidden states from every layer and mean-pooled layer 9. The diagnostic costs one extra pass but could save two if the stash drifts too far.

I'm deciding between a 256-sample diagnostic pass and reading everything at once. I'll read the first 1024 IDs, check how far the stash drifts, and only then do the rest, since that costs about the same as a diagnostic either way. Next I'll write stash.py and apply it.



Median LN gains look close enough that setting γ_S to them should keep the stash scale roughly stable. I'm writing stash.py to apply that to layers 0–8 from base weights, then checking the saved edits.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np\ndef build_stash(base, save_edit, c_s=4.0, gam_override=None):\n    cfg=np.load('stash_cfg.npz'); S=cfg['S']; J=cfg['J']; A0=cfg['A0']; b0=cfg['b0']; vmean=cfg['vmean']\n    F=np.load('feats.npz')\n    Mmean={l:F[f'M{l}'].astype(np.float64).mean(0) for l in range(10)}\n    for l in range(9):\n        L=f'bert.encoder.layer.{l}.'\n        g=lambda n: base(L+n).astype(np.float64)\n        Wq,bq,Wk,bk,Wv,bv=g('attention.self.query.weight'),g('attention.self.query.bias'),g('attention.self.key.weight'),g('attention.self.key.bias'),g('attention.self.value.weight'),g('attention.self.value.bias')\n        Wo,bo=g('attention.output.dense.weight'),g('attention.output.dense.bias')\n        W1,b1,W2,b2=g('intermediate.dense.weight'),g('intermediate.dense.bias'),g('output.dense.weight'),g('output.dense.bias')\n        g1,be1,g2,be2=g('attention.output.LayerNorm.weight'),g('attention.output.LayerNorm.bias'),g('output.LayerNorm.weight'),g('output.LayerNorm.bias')\n        if l==0:\n            Wq[:64]=0; bq[:64]=0\n            Wv[J]=A0; bv[J]=b0\n            bo+=Wo[:,J]@vmean[J]; Wo[:,J]=0\n            Wo[S,:]=0; bo[S]=0\n            for k in range(8): Wo[S[k],J[k]]=c_s\n        else:\n            hm=Mmean[l][S]\n            for W,b in [(Wq,bq),(Wk,bk),(Wv,bv)]:\n                b+=W[:,S]@hm; W[:,S]=0\n            Wo[S,:]=0; bo[S]=0\n        xm=Mmean[l+1][S]\n        b1+=W1[:,S]@xm; W1[:,S]=0\n        W2[S,:]=0; b2[S]=0\n        gs1=np.median(g1) if gam_override is None else gam_override[l][0]\n        gs2=np.median(g2) if gam_override is None else gam_override[l][1]\n        g1[S]=gs1; be1[S]=0; g2[S]=gs2; be2[S]=0\n        for n,a in [('attention.self.query.weight',Wq),('attention.self.query.bias',bq),('attention.self.key.weight',Wk),('attention.self.key.bias',bk),\n                    ('attention.self.value.weight',Wv),('attention.self.value.bias',bv),('attention.output.dense.weight',Wo),('attention.output.dense.bias',bo),\n                    ('intermediate.dense.weight',W1),('intermediate.dense.bias',b1),('output.dense.weight',W2),('output.dense.bias',b2),\n                    ('attention.output.LayerNorm.weight',g1),('attention.output.LayerNorm.bias',be1),('output.LayerNorm.weight',g2),('output.LayerNorm.bias',be2)]:\n            save_edit(L+n,a.astype(np.float32))\n'''\nopen('stash.py','w').write(code)\nimport importlib, sys; sys.path.insert(0,'.')\nimport stash; stash.build_stash(base, save_edit, c_s=4.0)\ne=edits(); print(len(e)); print([n for n in e if 'layer.0.' in n][:20])\n# sanity: layer 0 head0 query zero, Wo stash entries\ncfg=np.load('stash_cfg.npz'); S=cfg['S']; J=cfg['J']\nWo=W('bert.encoder.layer.0.attention.output.dense.weight'); print(Wo[S][:,J].round(2)); print(np.abs(Wo[S]).sum(1))"
}
```


## reasoning summary

Next I'm computing a5e_1792 from the upstream values and checking that its cls_hidden_L5 matches the one from pass 18. Then I'm weighing three L5 options: the full unpruned FFN as a ceiling, a base-only fit at m=1792 with lambda 10 and 30, and a mixed fit using a5e_1536 as in pass 18, to see whether it reproduces those logits.

I'm excluding the 64 ids used in the L0-2 least-squares data so the evaluation stays honest, and I'm also fitting a mixed model with a5e_1792 inside cross-validation.



I'm converting the probe read to exact L5 inputs, then comparing L5 options while excluding the 64 ids used in the L0–2 fits.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lsbuild import gelu, layer_data, build_l5_mixed\nfrom evalh import Lb, lab1, f1\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\ng1=base(pre+'attention.output.LayerNorm.weight').astype(np.float64); be1=base(pre+'attention.output.LayerNorm.bias').astype(np.float64)\nn=np.load('pass_19/cls_hidden_L6.npy').astype(np.float64)\nprint('n mean/var check', np.abs(n.mean(1)).max(), n.var(1).min(), n.var(1).max())\nae=n*g1+be1; np.save('a5e_1792.npy',ae.astype(np.float32))\nprint('L5 input same as pass18?', np.abs(np.load('pass_19/cls_hidden_L5.npy')-np.load('pass_18/cls_hidden_L5.npy')).max())\nab,W1,b1,U,b2=layer_data(5,base)\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\nD=base('classifier.dense.weight').astype(np.float64); bd=base('classifier.dense.bias'); O=base('classifier.out_proj.weight'); bo=base('classifier.out_proj.bias')\nhead=lambda h: np.tanh(h@D.T+bd)@O.T+bo\n# sanity: reproduce pass18 logits with mixed fit (a5e_1536)\nkeep,Un,b2n=build_l5_mixed(1792,base,'a5e_1536.npy',lam=30.0)\nGe=gelu(ae@W1.T+b1)\nh18=LN(ae+Ge@Un.T+b2n,g2,bb)\nprint('reproduce pass18 logits err', np.abs(head(h18)-np.load('pass_18/logits.npy')).max())\n"
}
```


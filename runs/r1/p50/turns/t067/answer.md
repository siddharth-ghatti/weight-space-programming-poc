## reasoning summary

Now I'm turning to L5, where the CLS FFN inputs from the edited and base models are both 1000 samples. The target for this layer is the final CLS output after the second layer norm, passed through the head to logits. I'm weighing a standard fit that matches the pruned FFN to the original FFN over the combined 2000 samples.

I'm weighing a second option for L5: refit the kept FFN units so that the LN2 input from the edited model matches the base model's, which would also correct upstream error. That means using only 1000 samples to fit a large matrix, so it risks overfitting. I'd keep it close to the standard solution with ridge, and I'm considering whether scoring lambda by logit agreement would need an FFN and head computation I'd rather not reimplement.

I'm checking the FFN+LN2+head computation as a local surrogate for choosing lambda, and planning to confirm it by reproducing the base model's final CLS from the base inputs first. If that matches, I'd compare the standard fit and the corrective fit on that basis, then verify the choice with real forward passes.

I'm working out the corrective target for L5: I want the edited model's pre-LayerNorm activation to match the base model's, so I'm defining the residual target and the constant that carries the second bias through.

I'm going to use ridge to pull the corrective fit toward the standard solution, so the adjustment D stays small and only the residual error is corrected. Now I'll compute it.



I'm checking that my local LN2+head reproduces the base logits, then fitting layer 5 against base agreement under 5-fold CV, comparing standard compensation with a corrective fit.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read()); exec(open('common.py').read())\ng1 = base(\"roberta.encoder.layer.5.attention.output.LayerNorm.weight\"); be1 = base(\"roberta.encoder.layer.5.attention.output.LayerNorm.bias\")\nz = np.load('pass_24/cls_hidden_L6.npy').astype(np.float64); a_ed = z*g1+be1\nnp.save('ffn_ed_L5_cls.npy', a_ed.astype(np.float32))\na_b = np.load('ffn_in_L5_cls.npy').astype(np.float64)\npre=\"roberta.encoder.layer.5.\"\nW1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\nW2 = base(pre+\"output.dense.weight\").astype(np.float64); b2 = base(pre+\"output.dense.bias\")\ng2 = base(pre+\"output.LayerNorm.weight\"); be2 = base(pre+\"output.LayerNorm.bias\")\ndef LN(x): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g2+be2\nWd = base('classifier.dense.weight'); bd = base('classifier.dense.bias'); Wo = base('classifier.out_proj.weight'); bo = base('classifier.out_proj.bias')\nhead = lambda C: np.tanh(C@Wd.T+bd)@Wo.T+bo\nHb = gelu(a_b@W1.T+b1); Zb = a_b + Hb@W2.T + b2\nprint('sanity base recompute vs BL max diff', np.abs(head(LN(Zb))-BL).max())\nHe = gelu(a_ed@W1.T+b1)\nreport('ed inputs, full L5 FFN', head(LN(a_ed + He@W2.T + b2)))\nnp.save('L5_Zb.npy', Zb)"
}
```


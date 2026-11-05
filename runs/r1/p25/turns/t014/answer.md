## reasoning summary

The full linear map only edges out the diagonal one (0.72 against 0.66), and the diagonal version folds completely into the preceding LayerNorm, so it stays a reasonable choice. I'm also wondering whether the R² computed in raw space is being driven by the few very large outlier dimensions in RoBERTa, since those would dominate a variance-weighted sum. I'll accept that for now. Next I'm setting up Experiment A.

I'm going with 6 layers at m=1536 for Experiment A, and I'm considering Experiment B as the other option, which drops layer 3 with the diagonal compensation folded into LN2 of layer 2. Looking at the parameter counts, 6L at m=1536 comes to about 68.1M, which is close to the target size. For A I'll record CLS hidden states from all layers and the logits, plus the mean hidden states so I can see how the token-level representations drift from the base model.



A full linear map only beats diagonal-affine slightly (R² 0.72 vs 0.66), so I'll use diagonal-affine as the compensation if a layer gets dropped. Next I'm running Experiment A: all 6 layers with the top 1536 FFN units each, folding pruned units' mean output into the bias, which gives 68.1M parameters.



## tool_use: run_python

```json
{
 "code": "def apply_ffn(m, layers=range(6), comp=True, tag=''):\n    keep={}\n    for l in layers:\n        d=np.load(f'imp_L{l}.npz')\n        k=np.sort(np.argsort(-d['I'])[:m]); keep[l]=k.tolist()\n        pr=np.setdiff1d(np.arange(3072),k)\n        U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n        b2=base(f'roberta.encoder.layer.{l}.output.dense.bias').astype(np.float64)\n        Eg=0.5*(d['Eg_in']+d['Eg_out'])\n        if comp: save_edit(f'roberta.encoder.layer.{l}.output.dense.bias',(b2+U[:,pr]@Eg[pr]).astype(np.float32))\n    return keep\nkeep=apply_ffn(1536)\nprint(set_structure(layers=list(range(6)),ffn_keep=keep))\nprint(edits())"
}
```


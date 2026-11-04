## reasoning summary

I'm noting that the top 1024 components carry about half the FFN's variance, so pruning down to that size would remove a lot of signal and likely distort the representations. The classifier gets refit afterward, though, which softens that. I'm also worried heavy FFN pruning could drag the deep features down to near shallow-model accuracy, around 97.4%. Another option to cut cost is dropping whole middle layers at full width, since each dropped layer saves about 7.09M parameters.

Next I'm weighing a test of FFN pruning at m=1024 across all twelve layers, and I'm considering m=1536 as well. The results will tell me whether to go smaller or larger. If m=1024 holds up, I'd try m=512 or m=256 next. If it falls to around 97.5%, I'd try m=2048 or drop layers instead.

I'm weighing whether one-config-per-pass is the real constraint, since zeroing could mimic non-uniform pruning for sensitivity analysis, though the final export needs a uniform width. Layer dropping is the other candidate, and I'm reconsidering it against FFN pruning with bias compensation, since the deeper features seem to need the full depth.

I'm planning Exp 1 as FFN pruning to m=1024 across all twelve layers, with reserved dimensions kept at zero and bias compensation for the dropped units. Before running it, I want to check label balance in the first 1024 examples and see how large the compensation terms come out.

I'm weighing the compensation term for the dropped units, where the bias shift depends on each unit's expected GELU output and its output weights. Under the Gaussian proxy those expectations come out small for units centred around -3, so the correction should stay modest. I'm also checking that the importance measure is the right one for this, and it looks fine since the compensation handles the mean.

I'm setting up the bias compensation for layers 0 through 10, leaving layer 11 alone since its FFN is switched off.



Pruning to 1024 units per FFN keeps only about half the proxy variance, so the representations will shift, but the head will be refitted on the pruned model's activations. Next I'll run the 12-layer pruned model over 1024 sentences to get the accuracy-by-depth profile and compare it paired against base.



## tool_use: run_python

```json
{
 "code": "import json, numpy as np\nP=practice(); y=np.array([LABELS.index(p['label']) for p in P])\nprint(np.bincount(y[:1024]))\nZ=np.load(WORK/'ffn_proxy.npz')\nRES=np.array([83,88,222,254,444,495,506,601])\ndef make_prune(m, layers=range(12)):\n    keep={}\n    for l in layers:\n        imp=Z[f'imp{l}']; mean=Z[f'mean{l}']\n        if l==11:\n            keep[l]=list(range(m)); continue\n        order=np.argsort(-imp); k=np.sort(order[:m]); drop=order[m:]\n        keep[l]=[int(i) for i in k]\n        W2=base(f'bert.encoder.layer.{l}.output.dense.weight'); b2=base(f'bert.encoder.layer.{l}.output.dense.bias')\n        comp=W2[:,drop]@mean[drop]\n        if l>=2: comp[RES]=0\n        save_edit(f'bert.encoder.layer.{l}.output.dense.bias', (b2+comp).astype(np.float32))\n        print(l,'comp norm',round(float(np.linalg.norm(comp)),3),'b2 norm',round(float(np.linalg.norm(b2)),3))\n    return keep\nkeep=make_prune(1024)\nn=set_structure(layers=list(range(12)), ffn_keep=keep)\nprint('params',n, structure()['m'])\n"
}
```


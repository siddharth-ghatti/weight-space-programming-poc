## reasoning summary

The importance scores are fairly evenly spread, with the top 768 units holding about half the summed std importance, though that's not quite the same as variance share, so I'll look at the squared-importance ratio instead. Layer 11's FFN output is all zeros, which fits with the FFN being off there. Next I'm setting up the experiment with m=768 units per layer across all 12 layers, folding the dropped units' expected GELU contribution into the output bias.

I'm weighing whether to try m=1024 or m=512, and settling on m=768 (25%) as a midpoint. At that setting, 10 layers come to about 59.9M, the same as 5 full layers, which is still heavy. If m=768 holds up well, I'd try 384 next. I also need to recompute the bias folding for each configuration, since it depends on which units are kept, so I'll write a helper that takes m and the per-layer keep lists and saves the bias edits.



Importance is spread fairly evenly, with the top 768 units holding about half of it, and the two proxies agree well through layers 1–8. Experiment B keeps all 12 layers at 768 units each, folding the dropped units into `output.dense.bias`, and next I'll run one read of 1,024 ids to see how each depth degrades at 25% FFN width.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, json\ndef apply_ffn_prune(m, layers, W, base, save_edit, set_structure, WORK, fold=True, imp_file='ffn_imp.npz', extra_keep=None):\n    F=np.load(WORK/imp_file)\n    keep={}\n    for L in layers:\n        imp=F[f'imp{L}']\n        if np.all(imp==0) or np.isnan(imp).any():\n            k=list(range(m))\n        else:\n            k=sorted(np.argsort(-imp)[:m].tolist())\n        keep[L]=k\n        if fold and L<=10:\n            E=F[f'E{L}']; Wout=W(f'bert.encoder.layer.{L}.output.dense.weight')\n            drop=np.setdiff1d(np.arange(3072), k)\n            b0=base(f'bert.encoder.layer.{L}.output.dense.bias')\n            save_edit(f'bert.encoder.layer.{L}.output.dense.bias', (b0 + Wout[:,drop]@E[drop]).astype(np.float32))\n    return set_structure(layers=layers, ffn_keep=keep)\n'''\n(WORK/'prunelib.py').write_text(code)\nimport sys; sys.path.insert(0,str(WORK))\nfrom prunelib import apply_ffn_prune\nprint(apply_ffn_prune(768, list(range(12)), W, base, save_edit, set_structure, WORK))\nprint(edits())\n"
}
```


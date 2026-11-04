## reasoning summary

I'm setting up experiment A: layers 0 through 10, keeping the top 1024 units by std score and compensating the mean into output.dense.bias. I'll screen on the first 1024 ids in one pass and compare against base features restricted to the same set, though I'm also wondering whether the full 1496 would make the comparison fairer.

I'm weighing whether to include the bias compensation, and I'm fairly sure it's reasonable since it's a standard technique. Next I'll write a helper, build_ffn_edits, that takes the layers, the number of units, the scoring method, and a compensation flag, saves the output.dense.bias edits for the kept layers, and then calls set_structure.



The harness needs m ≥ 1, so I'm adding a helper that keeps the top-m units per layer and folds the dropped units' mean output into the dense bias. Experiment A (layers 0–10 at m=1024) is next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nadd = r'''\ndef ffn_keep_sets(layers, m, kind='std'):\n    S=np.load('ffn_scores.npz'); return {l: sorted(np.argsort(-S[f'{kind}_{l}'])[:m].tolist()) for l in layers}\ndef apply_ffn_prune(layers, m, comp=True, kind='std', base=None, save_edit=None, drop_edit=None, edits=None):\n    S=np.load('ffn_scores.npz'); keep=ffn_keep_sets(layers,m,kind)\n    for l in layers:\n        nm=f'bert.encoder.layer.{l}.output.dense.bias'\n        b=base(nm).copy()\n        if comp:\n            Wo=base(f'bert.encoder.layer.{l}.output.dense.weight'); rem=np.setdiff1d(np.arange(3072),keep[l])\n            b=b+Wo[:,rem]@S[f'mean_{l}'][rem]\n        save_edit(nm,b.astype(np.float32))\n    return keep\n'''\nopen('lib.py','a').write(add)\nfrom lib import *\nkeep=apply_ffn_prune(list(range(11)),1024,base=base,save_edit=save_edit)\nprint(set_structure(layers=list(range(11)), ffn_keep=keep))\nprint(len(edits()))"
}
```


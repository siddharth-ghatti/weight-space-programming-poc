## reasoning summary

I'm going to merge everything into one npz, then run ridge regression per layer with standardization, leaving out the reserved dimensions. I'll use stratified 5-fold CV, repeated across several seeds. The "variant" and "line" fields might mean near-duplicate sentences, so grouped CV could be the better choice. Let me check the metadata counts for variants, lines, and registers.



Both reads are done; I'm merging them and running 5-fold ridge CV on each layer's mean and [CLS] features, excluding the 8 reserved dimensions that were fitted on this practice set. I'll also check the metadata fields for near-duplicate variants.



## tool_use: run_python

```json
{
 "code": "import json, numpy as np, collections\nP = practice(); ids=[p['id'] for p in P]\nm1 = json.load(open(WORK/'pass_1/meta.json')); m2=json.load(open(WORK/'pass_2/meta.json'))\nprint(m1.keys())\norder = (m1.get('ids') or []) + (m2.get('ids') or [])\nprint(order[:3], len(order), order==ids)\nD={}\nfor L in range(13):\n    for kind in ['cls_hidden','mean_hidden']:\n        D[f'{kind}_L{L}'] = np.concatenate([np.load(WORK/f'pass_1/{kind}_L{L}.npy'), np.load(WORK/f'pass_2/{kind}_L{L}.npy')])\nD['pooled']=np.concatenate([np.load(WORK/'pass_1/pooled.npy'),np.load(WORK/'pass_2/pooled.npy')])\nD['logits']=np.concatenate([np.load(WORK/'pass_1/logits.npy'),np.load(WORK/'pass_2/logits.npy')])\ny = np.array([LABELS.index(p['label']) for p in P])\nD['y']=y\nnp.savez(WORK/'base_feats.npz', **D)\nprint('base acc on practice', (D['logits'].argmax(1)==y).mean())\nprint(collections.Counter(p['variant'] for p in P)); print(collections.Counter(p['register'] for p in P)); print(collections.Counter(p['line'] for p in P))\n"
}
```


## reasoning summary

I'm going to combine the results and save them as base_mean.npz, then run cross-validated ridge regression for each layer. First I need to identify the 8 reserved dimensions, the ones where the layer 1 output layernorm weight changed, so I can exclude them. For the ridge setup I'm weighing one-vs-all regression on centered one-hot targets, possibly with standardized features, using 5-fold stratified CV over several lambda values.



I'm merging the two reads into one array and excluding the 8 reserved channel dims, since they leak ridge scores. Then I'll run 5-fold CV ridge accuracy for each layer's mean and [CLS] features.



## tool_use: run_python

```json
{
 "code": "import json\nP = practice(); lab = {x['id']:LABELS.index(x['label']) for x in P}\nm1=json.load(open('pass_1/meta.json')); m2=json.load(open('pass_2/meta.json'))\nprint(m1.keys())\nids1 = m1.get('ids'); ids2=m2.get('ids')\nallids = ids1+ids2\ny = np.array([lab[i] for i in allids])\nD={}\nfor L in range(13):\n    for kind in ['mean_hidden','cls_hidden']:\n        D[f'{kind}_L{L}'] = np.concatenate([np.load(f'pass_1/{kind}_L{L}.npy'), np.load(f'pass_2/{kind}_L{L}.npy')])\nnp.savez(WORK/'base_reads.npz', y=y, ids=np.array(allids), **D)\ng = W('bert.encoder.layer.1.output.LayerNorm.weight'); g0=base('bert.encoder.layer.1.output.LayerNorm.weight')\nres = np.where(g==0)[0]; print('reserved dims', res)\nnp.save(WORK/'reserved.npy', res)\n"
}
```


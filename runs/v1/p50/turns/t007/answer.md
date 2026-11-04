## reasoning summary

I'm combining everything into single arrays and saving them as feats.npz. Then I'll run ridge and LDA probes per layer for both mean and CLS features, using 5-fold stratified CV. I also need to check meta.json to see whether the mean pooling includes CLS and SEP tokens.



Both reads are finished; next I'll merge them per layer, confirm how the mean is defined, and run ridge and shrinkage-LDA probes with 5-fold CV.



## tool_use: run_python

```json
{
 "code": "import json\nm1 = json.load(open(WORK/'pass_1/meta.json'))\nprint({k:(v if not isinstance(v,list) else v[:3]) for k,v in m1.items()})\nP = practice()\nlab = np.array([LABELS.index(p['label']) for p in P])\nnp.save(WORK/'labels.npy', lab)\nfor kind in ['mean','cls']:\n    arr = np.stack([np.concatenate([np.load(WORK/f'pass_1/{kind}_hidden_L{l}.npy'), np.load(WORK/f'pass_2/{kind}_hidden_L{l}.npy')]) for l in range(13)])\n    np.save(WORK/f'base_{kind}.npy', arr)\n    print(kind, arr.shape)\nprint(LABELS)\n"
}
```


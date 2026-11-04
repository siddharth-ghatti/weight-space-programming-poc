## reasoning summary

I'm comparing the model's logits to the offline values, which come out at about 0.3 times as large, so there's probably a scale factor or a layer-norm difference, with a residual of around 0.15 left unexplained. Pass 14 had reported a correlation above 0.9999, but now it's only 0.98, so the offline logits may have been computed differently then. I'm about to check the ratio for p0913.

The ratios for the three items come out nearly the same, around 0.23, so each item seems to have its own scale factor, likely from layer-norm dividing by a per-item standard deviation. That scale would not change the argmax, which fits. I still need to confirm that the proportionality holds for each item.



The model's logits appear to be the offline logits scaled by a positive per-item factor, about 0.23 for p0913. That likely comes from LayerNorm's per-sentence scaling, which doesn't change the argmax, so I'm checking this for each item.



## tool_use: run_python

```json
{
 "code": "\nimport json, numpy as np\nP=practice(); idx={r['id']:i for i,r in enumerate(P)}\nfor k in (14,15):\n    p=json.load(open(f'pass_{k}/predictions.json'))\n    off=np.load('offline_logits_S.npy')\n    L=np.array([it['logits_edited'] for it in p]); O=off[[idx[it['id']] for it in p]]\n    a=(L*O).sum(1)/(O*O).sum(1)\n    res=np.abs(L-a[:,None]*O).max()\n    pc=[np.corrcoef(L[i],O[i])[0,1] for i in range(len(L))]\n    print(k,'scale range',a.min(),a.max(),'max resid',res,'min per-item corr',min(pc))\n"
}
```


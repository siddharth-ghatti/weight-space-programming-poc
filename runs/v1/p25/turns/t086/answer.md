## reasoning summary

I've got one error inside the sample, and I'm wondering whether the offline model also made training errors. I'll compare it against the offline logits to check.



The model gets 511 of 512 on the longest sentences, and I'm checking whether the offline fit misses that same one and whether the logits match the offline computation.



## tool_use: run_python

```json
{
 "code": "\nimport json, numpy as np\nP=practice(); idx={r['id']:i for i,r in enumerate(P)}\np=json.load(open('pass_15/predictions.json'))\noff=np.load('offline_logits_S.npy'); print(off.shape)\nL=np.array([it['logits_edited'] for it in p]); O=off[[idx[it['id']] for it in p]]\nprint('corr', np.corrcoef(L.ravel(),O.ravel())[0,1], 'maxabs diff', np.abs(L-O).max(), 'scale', np.abs(O).max())\nprint('argmax agree', (L.argmax(1)==O.argmax(1)).mean())\ny=np.array([LABELS.index(P[i]['label']) if isinstance(P[i]['label'],str) else P[i]['label'] for i in range(len(P))])\nprint('offline in-sample acc all', (off.argmax(1)==y).mean())\nfor it in p:\n    if it['pred_edited']!=it['gold']: print(it['id'], it['gold'], it['pred_edited'], it['text'])\n"
}
```


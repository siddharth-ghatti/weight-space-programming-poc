## tool_use: run_python

```json
{
 "code": "import json, numpy as np\nP=json.load(open('pass_14/predictions.json')); off=np.load('offline_logits_S.npy')\nidx=np.array([int(p['id'][1:]) for p in P]); Lm=np.array([p['logits_edited'] for p in P])\nLo=off[idx]\n# model logits = tanh(scaled offline), compare via correlation after centering per row and argmax\nagree=(Lm.argmax(1)==Lo.argmax(1)).mean()\nr=[np.corrcoef(np.arctanh(np.clip(Lm[i],-.999,.999))-np.arctanh(np.clip(Lm[i],-.999,.999)).mean(), Lo[i]-Lo[i].mean())[0,1] for i in range(len(P))]\nprint('argmax agree',agree,'min corr',np.min(r),'median corr',np.median(r))\n# margin check\nsrt=np.sort(Lm,1); print('min margin (model)',(srt[:,-1]-srt[:,-2]).min())"
}
```


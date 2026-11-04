## tool_use: run_python

```json
{
 "code": "import json\np=json.load(open('pass_8/predictions.json'))\nids=json.load(open('ids_order.json')); idx={s:i for i,s in enumerate(ids)}\nexp=np.load('v2_expected_logits.npy')\nL=np.array([r['logits_edited'] for r in p]); E=exp[[idx[r['id']] for r in p]]\nprint(np.abs(L-E).max(), np.corrcoef(L.ravel(),E.ravel())[0,1])\nprint(L[:3]); print(E[:3])\nsrt=np.sort(L,1); print('min margin', (srt[:,-1]-srt[:,-2]).min())\n"
}
```


## tool_use: run_python

```json
{
 "code": "import os, json; os.chdir(WORK)\nimport numpy as np\nitems=json.load(open('pass_18/predictions.json'))\nids=json.load(open('ids_order.json')); pos={i:k for k,i in enumerate(ids)}\nL=np.load('v6_expected_logits.npy')\ngot=np.array([it['logits_edited'] for it in items]); exp=L[[pos[it['id']] for it in items]]\nprint('max abs err', np.abs(got-exp).max(), 'argmax agree', (got.argmax(1)==exp.argmax(1)).mean())\n# margin stats\ns=np.sort(got,1); print('min margin', (s[:,-1]-s[:,-2]).min())\n"
}
```


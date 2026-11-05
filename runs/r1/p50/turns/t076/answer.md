## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\ndef load_pred(d, key='edited'):\n    P = json.load(open(f'{d}/predictions.json')); lg = np.zeros((1000,4)); seen=np.zeros(1000,bool)\n    for r in P:\n        i = int(r['id'][1:]); lg[i] = r[f'logits_{key}']; seen[i]=True\n    return lg, seen\nlg, seen = load_pred('pass_25'); print(seen.sum())\nnp.save('lg_m1536_v2.npy', lg)\nreport('m1536 v2', lg)\ns = np.sort(BL,1); marg = s[:,-1]-s[:,-2]; top=bp; ru=np.argsort(BL,1)[:,-2]\nfor name,L in [('m1536 old',np.load('pass_10/logits.npy')),('m2048 old',np.load('pass_12/logits.npy')),('m1536 v2',lg)]:\n    dm=L-BL; nm = dm[np.arange(1000),top]-dm[np.arange(1000),ru]; p=L.argmax(1); fl=p!=bp\n    print(name, 'flips', fl.sum(), 'broken', (fl&(bp==y)).sum(), 'fixed', (fl&(p==y)).sum(), 'noise std', nm.std().round(3), 'logit rms', np.sqrt((dm**2).mean()).round(3))"
}
```


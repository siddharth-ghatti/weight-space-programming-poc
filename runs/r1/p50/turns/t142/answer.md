## tool_use: run_python

```json
{
 "code": "import json\nexec(open('risk.py').read())\npr=json.load(open('pass_31/predictions.json')); idx={r['id']:i for i,r in enumerate(pr)}\nlg=np.array([pr[idx[i]]['logits_edited'] for i in ids]); np.save('lg_pass31_m1792v2.npy',lg)\nM={'m1536v1':np.load('lg_pass25.npy'),'m1792v2':lg,'m2048v2':np.load('lg_pass30_m2048v2.npy')}\nfor n,l in M.items():\n    print(n,'agr',(l.argmax(1)==bl.argmax(1)).mean(),'rms %.4f'%np.sqrt(((l-bl)**2).mean()))\nshow(risk(M))\n"
}
```


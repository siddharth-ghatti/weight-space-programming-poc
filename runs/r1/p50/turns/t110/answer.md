## message

Now D2: restore L0–1 to original and apply the v2 fits to L2–4 only.

## tool_use: run_python

```json
{
 "code": "import shutil\nexec(open('common.py').read())\nids=json.load(open('all_ids.json')); idx={i:k for k,i in enumerate(ids)}\nlg=np.zeros((1000,4))\nfor r in json.load(open('pass_27/predictions.json')): lg[idx[r['id']]]=r['logits_edited']\nnp.save('lg_D1_L01.npy',lg); report('D1 L0-1 pruned', lg); print(' mean|dl| %.4f rms %.4f'%(np.abs(lg-BL).mean(), np.sqrt(((lg-BL)**2).mean())))\nfor L in [0,1]:\n    for s in ['output.dense.weight','output.dense.bias']: drop_edit(f'roberta.encoder.layer.{L}.{s}')\nfor L in [2,3,4]:\n    for s in ['output.dense.weight','output.dense.bias']:\n        n=f'roberta.encoder.layer.{L}.{s}'; save_edit(n, np.load(f'edits_m1536_v2/{n}.npy'))\nprint(sorted(edits()))\n"
}
```


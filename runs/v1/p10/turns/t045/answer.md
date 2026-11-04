## reasoning summary

I'm checking that the training accuracy, which is measured in-sample, lines up with the logits from the simulation.



The pipeline reaches 99.8% on 512 in-sample sentences, which is training accuracy rather than a generalization estimate. Next I'm checking that the harness logits match my simulated ones numerically.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\npr=json.load(open('pass_9/predictions.json'))\nprint(type(pr), (list(pr.keys()) if isinstance(pr,dict) else pr[0].keys()))\nitems = pr if isinstance(pr,list) else pr.get('items', pr.get('predictions'))\nprint(items[0])\nids=[p['id'] for p in practice()]; idx={k:i for i,k in enumerate(ids)}\nsim=np.load('sim_logits_v3.npy')\ndiffs=[]\nfor it in items:\n    i=idx[it['id']]\n    lg=np.array(it['logits'] if 'logits' in it else it['edited_logits'])\n    diffs.append(np.abs(lg-sim[i]).max())\nprint('max abs diff', max(diffs), 'mean', np.mean(diffs))"
}
```


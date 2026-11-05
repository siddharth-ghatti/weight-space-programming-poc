## reasoning summary

Sequential m1024 came out worse on both agreement and accuracy than the earlier m1024 run, even though L5 was at m1024 in both cases. Next I'm checking agreement for pass 25 at m1536.



## message

The sequential m=1024 model only matches base 89.6% of the time (accuracy .849), which isn't good enough. Now pass 25 (m=1536):

## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nids=json.load(open('all_ids.json')); idx={i:k for k,i in enumerate(ids)}\npr=json.load(open('pass_25/predictions.json'))\nlg=np.zeros((1000,4)); \nfor r in pr: lg[idx[r['id']]]=r['logits_edited']\nreport('m1536 pass25', lg)\nnp.save('lg_pass25.npy', lg)\ndm = lg - BL\nprint('mean|dlogit|', np.abs(dm).mean())\n# margin of base\ns=np.sort(BL,1); marg=s[:,-1]-s[:,-2]\nflip = lg.argmax(1)!=bp\nprint('flips', flip.sum(), 'base margins of flips', np.round(np.sort(marg[flip]),2))\nlg2=np.load('lg_m1536_v2.npy'); print(lg2.shape); report('lg_m1536_v2', lg2)\nprint(json.load(open('keep_sets.json')).keys(), [len(v) for v in json.load(open('keep_sets.json')).values()])\n"
}
```


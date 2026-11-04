## reasoning summary

The pipeline is confirmed. I'm weighing whether to push past the 97.8% out-of-fold score; in-sample accuracy is 1.0, which looks overfit, so the OOF figure is the honest estimate. More layers or token-level features might help, but the gains look marginal and risky. I'll compare the expected predictions against the pass_12 logits using expected_v3.npy, update the notes, and submit.

Next I'll check the logits against expected_v3 for those 472 items.



The harness predict path matches on all 472 held-out sentences, though those sentences were in the fitting data, so it confirms correctness rather than generalization. I'm checking the harness logits against my numpy expectation next.



## tool_use: run_python

```json
{
 "code": "\nimport json\npr=json.load(open('pass_12/predictions.json'))\nprint(type(pr), (list(pr.keys()) if isinstance(pr,dict) else pr[0].keys()))\nitems = pr['edited'] if isinstance(pr,dict) and 'edited' in pr else (pr.get('items') if isinstance(pr,dict) else pr)\nprint(type(items), (items[0] if isinstance(items,list) else list(items.keys())[:5]))\nE=np.load('expected_v3.npy'); print(E.shape)\nids=json.load(open('ids.json')); print(len(ids), ids[:3])\n"
}
```


## reasoning summary

The logits are small, so tanh stays in its near-linear range and nothing saturates, and the smallest margin of 0.0033 is still fine given float32 precision. The classifier is an identity mapping on pooled dimensions 0 through 7. I'm now wondering whether the logits were rounded to four decimals, which the margin would suggest, but that doesn't change the conclusion. With plenty of budget left, I'm considering whether to attempt an improvement over the kernel readout.

I'm weighing whether the remaining passes could buy a real gain over the current readout, and the kernel and linear versions are within noise of each other, so I'm leaning toward submitting. One option I'm considering is attention-based pooling with a learned query, fit in closed form from token-level states.

I'm weighing whether a final verification pass is worth one of my remaining passes. The earlier offline logits matched closely, but I haven't checked the longer practice sentences, so I'll run the edited model on the longest ones not already covered by pass 14.



The pooled values are in tanh's near-linear range, so saturation can't create ties, and I'll check the edited model on the 512 longest practice sentences before submitting.



## tool_use: run_python

```json
{
 "code": "\nimport json, numpy as np\nP=practice(); seen={it['id'] for it in json.load(open('pass_14/predictions.json'))}\nrest=[r for r in P if r['id'] not in seen]\ntoks=tokenize([r['text'] for r in rest]); lens=[len(t['ids']) for t in toks]\norder=np.argsort(lens)[::-1][:512]\nids=[rest[i]['id'] for i in order]\nprint(len(rest), max(lens), np.array(lens)[order].min(), np.mean(lens))\njson.dump(ids, open('ids_long.json','w'))\nprint(json.dumps(ids))\n"
}
```


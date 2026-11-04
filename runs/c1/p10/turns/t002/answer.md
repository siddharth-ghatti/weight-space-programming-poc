## reasoning summary

Starting fresh, I'm going to look at the data first, checking the tokenization lengths and which dimensions are reserved.



## message

This is a fresh start. Next I'll look at the data (lengths), find the 8 reserved dimensions from the layer-1 LayerNorm edits, and check a few structural details of the start weights.

## tool_use: run_python

```json
{
 "code": "P = practice()\nprint(len(P), P[0])\ntoks = tokenize([p['text'] for p in P])\nL = [len(t['ids']) for t in toks]\nprint('len stats', np.min(L), np.median(L), np.max(L))\nimport collections\nprint(collections.Counter(p['label'] for p in P))\n# reserved dims\nw1 = base('bert.encoder.layer.1.output.LayerNorm.weight'); b1 = base('bert.encoder.layer.1.output.LayerNorm.bias')\nres = np.where((w1==0)&(b1==0))[0]; print('reserved', res, w1[res], b1[res])\ncw = base('classifier.weight'); cb = base('classifier.bias')\nprint(np.nonzero(cw)[0][:20], np.nonzero(cw)[1][:20], cw[np.nonzero(cw)], cb)\nfor l in [2,6,10]:\n    qb = base(f'bert.encoder.layer.{l}.attention.self.query.bias'); qw = base(f'bert.encoder.layer.{l}.attention.self.query.weight')\n    zero_heads = [h for h in range(12) if np.abs(qw[h*64:(h+1)*64]).sum()==0]\n    print(l, 'zero-query heads', zero_heads)\n"
}
```


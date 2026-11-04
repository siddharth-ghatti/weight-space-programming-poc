## reasoning summary

I'm starting fresh here, so first I need to pin down which dimensions are reserved, then I'll look through the data.



## message

Fresh start. Next I'll find the 8 reserved dimensions (the ones whose layer-1 output LayerNorm was zeroed), check how the pooler and classifier are wired, and look at sentence lengths.

## tool_use: run_python

```json
{
 "code": "g = base('bert.encoder.layer.1.output.LayerNorm.weight'); b = base('bert.encoder.layer.1.output.LayerNorm.bias')\nres = np.where(np.abs(g) < 1e-8)[0]; print('reserved', res, g[res], b[res])\ncw = base('classifier.weight'); print(np.nonzero(cw)); print(cw[np.nonzero(cw)], base('classifier.bias'))\nP = practice(); print(len(P), P[0])\ntoks = tokenize([p['text'] for p in P]); L = np.array([len(t['ids']) for t in toks]); print(L.min(), L.mean(), L.max())\nfor l in [2,6,10]:\n    q = base(f'bert.encoder.layer.{l}.attention.self.query.weight'); \n    z = [h for h in range(12) if np.abs(q[h*64:(h+1)*64]).max()==0]; print(l, 'zero-query heads', z)\nopen(WORK/'notes.md','w').write(f\"# Notes\\n- reserved dims: {res.tolist()}\\n- carry heads layers 2,6,10\\n\")"
}
```


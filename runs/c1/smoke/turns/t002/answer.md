## reasoning summary

I'm going to inspect the weights next, starting with the reserved dimensions, the carry heads, and the layer 11 weights.



## tool_use: run_python

```json
{
 "code": "\n# Which dims are reserved? compare LN at layer 1 output vs. known bert? we only have start model. look for zeros.\nw=base('bert.encoder.layer.1.output.LayerNorm.weight'); b=base('bert.encoder.layer.1.output.LayerNorm.bias')\nres=np.where(w==0)[0]; print('reserved',res, b[res])\n# layer 11 structure\nfor n in ['attention.self.query','attention.self.key','attention.self.value','attention.output.dense']:\n    A=base(f'bert.encoder.layer.11.{n}.weight'); bb=base(f'bert.encoder.layer.11.{n}.bias')\n    print(n, np.abs(A).max(), (np.abs(A)>0).sum(), np.abs(bb).max())\nprint('l11 ffn out', np.abs(base('bert.encoder.layer.11.output.dense.weight')).max(), np.abs(base('bert.encoder.layer.11.output.dense.bias')).max())\nprint('l11 LN', base('bert.encoder.layer.11.attention.output.LayerNorm.weight')[:8], base('bert.encoder.layer.11.output.LayerNorm.weight')[:8], base('bert.encoder.layer.11.output.LayerNorm.bias')[:8])\n# carry head layer 2\nfor L_ in [2,6,10]:\n    q=base(f'bert.encoder.layer.{L_}.attention.self.query.weight'); \n    print(L_,'q zero rows', np.where(np.abs(q).sum(1)==0)[0][:80])\n"
}
```


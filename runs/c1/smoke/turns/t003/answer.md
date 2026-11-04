## reasoning summary

I'm working through layer 11, where the query weights and bias are all zero, so attention comes out uniform across every head. The value and output projections both carry near-identity weights, so each token ends up with roughly twice the average plus its own residual, and I'm checking whether that holds for the CLS token too.

I'm checking the layer 11 LayerNorm next: its weight is 1 and bias is 0, so it's just normalization after the residual, and the same holds for the output norm. Then I'll look at the pooler weights, rows 0 through 7.



## tool_use: run_python

```json
{
 "code": "\nres=[83,88,222,254,444,495,506,601]\npw=base('bert.pooler.dense.weight'); pb=base('bert.pooler.dense.bias')\nprint('pooler nonzero rows', np.where(np.abs(pw).sum(1)>0)[0], pb[:10], np.abs(pw[8:]).max())\ncw=base('classifier.weight'); print(np.nonzero(cw))\nprint(np.abs(pw[:8]).sum(0)[res], np.abs(pw[:8]).max())\n# carry head details layer 2\nfor L_,h in [(2,0),(6,11),(10,10)]:\n    p=f'bert.encoder.layer.{L_}.attention.'\n    q=base(p+'self.query.weight'); qb=base(p+'self.query.bias')\n    v=base(p+'self.value.weight'); vb=base(p+'self.value.bias')\n    o=base(p+'output.dense.weight'); ob=base(p+'output.dense.bias')\n    sl=slice(h*64,(h+1)*64)\n    print(L_, 'q head max', np.abs(q[sl]).max(), np.abs(qb[sl]).max(), 'v head nonzero rows', np.where(np.abs(v[sl]).sum(1)>0)[0], 'vb', np.abs(vb[sl]).max())\n    print('  o cols nonzero rows', np.where(np.abs(o[:,sl]).sum(1)>0)[0], 'ob',ob[res])\n    print('  LN attn w/b res', base(p+'output.LayerNorm.weight')[res], base(p+'output.LayerNorm.bias')[res])\n    print('  LN out w/b res', base(f'bert.encoder.layer.{L_}.output.LayerNorm.weight')[res], base(f'bert.encoder.layer.{L_}.output.LayerNorm.bias')[res])\n    print('  k head', np.abs(base(p+'self.key.weight')[sl]).max())\n"
}
```


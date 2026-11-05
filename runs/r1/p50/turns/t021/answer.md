## reasoning summary

I'm saving the layer 5 CLS FFN inputs as a 1000×768 array and removing the probe edits so the full model structure is back. Next I'll analyze the FFN units in each layer, computing their activations with exact erf GELU, scoring each unit by its contribution energy, and checking how much of the output variance the top units account for.

I'm checking how the relative error changes as I vary the number of retained units, comparing a mean-compensated version with a ridge least-squares version fit on held-out tokens. The error is measured against the residual stream a + f(a), since LayerNorm makes that the relevant scale.



## message

Saving the layer-5 `<s>` FFN inputs (1000 rows), removing all probe edits and restoring the full structure. Then I'll analyse each layer's FFN units offline.

## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'probe.py').read())\nZ = np.load(WORK/'pass_9/cls_hidden_L6.npy')\nprint('mean', np.abs(Z.mean(1)).max(), 'var', Z.var(1).min(), Z.var(1).max())\ng = base(\"roberta.encoder.layer.5.attention.output.LayerNorm.weight\"); b = base(\"roberta.encoder.layer.5.attention.output.LayerNorm.bias\")\nnp.save(WORK/'ffn_in_L5_cls.npy', (Z*g+b).astype(np.float32))\nprobe_off(5); clear_structure()\nprint(edits(), structure()['layers'], structure()['m'])"
}
```


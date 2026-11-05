## reasoning summary

I'm checking that the normalized token hidden states have roughly zero mean and unit variance per token. Next I'll store the affine output for the masked tokens using the layer 1 scale and shift, then set up the probe for layer 1.



## message

Checking that the probe output really is normalized (mean ≈ 0, var ≈ 1 per token), saving the reconstructed FFN inputs for layer 0, then switching the probe to layer 1.

## tool_use: run_python

```json
{
 "code": "import json\nexec(open(WORK/'probe.py').read())\ndef save_probe(L, passdir, hidx):\n    Z = np.load(WORK/passdir/f'token_hidden_L{hidx}.npy'); M = np.load(WORK/passdir/'attention_mask.npy').astype(bool)\n    z = Z[M]\n    print('mean', np.abs(z.mean(1)).max(), 'var', z.var(1).min(), z.var(1).max(), z.shape)\n    g = base(f\"roberta.encoder.layer.{L}.attention.output.LayerNorm.weight\"); b = base(f\"roberta.encoder.layer.{L}.attention.output.LayerNorm.bias\")\n    a = z*g + b\n    # also record which rows are CLS\n    iscls = np.zeros(M.shape, bool); iscls[:,0]=True\n    np.save(WORK/f'ffn_in_L{L}.npy', a.astype(np.float32)); np.save(WORK/f'ffn_in_L{L}_iscls.npy', iscls[M])\nsave_probe(0, 'pass_4', 1)\nprobe_off(0)\nprobe_on(1); print(edits(), structure()['layers'])"
}
```


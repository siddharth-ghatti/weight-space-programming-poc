## reasoning summary

I'm checking that n has a per-token mean near zero and variance near one. Once that holds, I'll compute a from γ1 and β1, save the masked-token values as true_a_L0.npy, and then set up the probe for layer 1.



## message

Checking that the probe output is properly normalized (per-token mean ≈ 0, variance ≈ 1), saving the true layer-0 FFN input, and setting up the layer-1 probe:

## tool_use: run_python

```json
{
 "code": "def save_probe(l, pd):\n    mask=np.load(f'{pd}/attention_mask.npy').astype(bool)\n    n=np.load(f'{pd}/token_hidden_L{l+1}.npy')[mask].astype(np.float64)\n    print('per-token mean |max|',np.abs(n.mean(1)).max(),' var range',n.var(1).min(),n.var(1).max())\n    g=base(f'roberta.encoder.layer.{l}.attention.output.LayerNorm.weight').astype(np.float64)\n    b=base(f'roberta.encoder.layer.{l}.attention.output.LayerNorm.bias').astype(np.float64)\n    a=n*g+b\n    np.save(f'true_a_L{l}.npy',a.astype(np.float32))\n    # compare to proxies\n    Hin=np.load(f'pass_2/token_hidden_L{l}.npy')[mask].astype(np.float64); Hout=np.load(f'pass_2/token_hidden_L{l+1}.npy')[mask].astype(np.float64)\n    den=((a-a.mean(0))**2).sum()\n    print(f'L{l}: R2 of proxy in {1-((a-Hin)**2).sum()/den:.3f}, proxy out {1-((a-Hout)**2).sum()/den:.3f}, avg {1-((a-(Hin+Hout)/2)**2).sum()/den:.3f}')\nsave_probe(0,'pass_5')\nsetup_probe(1)"
}
```


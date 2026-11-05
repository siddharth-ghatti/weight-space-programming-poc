## reasoning summary

Before running the full pass, I want to sanity-check the linear units on a sample of true inputs, confirming that the pre-activation z_k stays above 4 across all tokens. I'm only checking the minimum for the linear-unit rows, since this is just verifying the linear regime from the closed-form math.



## message

Before spending a pass, I'll check that the linear units stay in GELU's linear regime (pre-activation well above 0) on the true FFN inputs.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nst=structure()\nfor l in range(6):\n    a=np.load(f'true_a_L{l}.npy') if l<5 else np.load('true_a5_cls.npy')\n    W1=W(f'roberta.encoder.layer.{l}.intermediate.dense.weight'); b1=W(f'roberta.encoder.layer.{l}.intermediate.dense.bias')\n    lin=np.where(b1>14)[0]\n    z=a@W1[lin].T+b1[lin]\n    print(l,len(lin),'min z',z.min().round(2),'pct<5',(z<5).mean().round(5),'std',z.std(0).mean().round(2))"
}
```


## message

Next I'll save the layer-1 inputs and set up the layer-2 probe.

## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nexec(open('probe.py').read())\nmask=np.load('pass_15/attention_mask.npy').astype(bool)\nn=np.load('pass_15/token_hidden_L2.npy')[mask].astype(np.float64)\nprint('var range',n.var(1).min(),n.var(1).max(), mask.sum())\nl=1\ng=base(f'roberta.encoder.layer.{l}.attention.output.LayerNorm.weight').astype(np.float64)\nb=base(f'roberta.encoder.layer.{l}.attention.output.LayerNorm.bias').astype(np.float64)\nnp.save(f'true2_a_L{l}.npy',(n*g+b).astype(np.float32))\nprint(setup_probe(2, save_edit, drop_edit, edits, set_structure))\n"
}
```

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0647",
  "p0664",
  "p0697",
  "p0537",
  "p0533",
  "p0679",
  "p0762",
  "p0513",
  "p0694",
  "p0132",
  "p0494",
  "p0573",
  "p0563",
  "p0597",
  "p0463",
  "p0388",
  "p0572",
  "p0977",
  "p0249",
  "p0318",
  "p0465",
  "p0552",
  "p0427",
  "p0601",
  "p0464",
  "p0596",
  "p0447",
  "p0952",
  "p0781",
  "p0230",
  "p0052",
  "p0591"
 ],
 "layers": [
  3
 ],
 "outputs": [
  "token_hidden"
 ]
}
```


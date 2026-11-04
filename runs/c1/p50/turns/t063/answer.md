## reasoning summary

I'm weighing whether to sweep all nine layers with separate passes, and I'm leaning toward running them in parallel so I can try several values of m at once.

I'm deciding whether to expose layers 0 through 8 or extend to 9, with about 41 passes left in budget. I'll start with 0 to 8 and only add layer 9 if I need it, and I'm setting up a helper that zeroes the attention and FFN outputs so each exposure pass can simply be dropped after the read.

I'm checking the exposure trick for layer l, and I don't see a way to expose the FFN input without breaking downstream layers, so I'll stick with the one-pass-per-layer setup. I'm going to write the helper now and start with layer 0.



The exposure set is 64 sentences (3,459 tokens), and I'm writing the helper that installs exposure edits for layer l, starting at layer 0. Layers 0–9 have no saved edits, so I can drop the exposure edits afterwards.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, json\nEXPO_NAMES = ['attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','output.dense.weight','output.dense.bias','output.LayerNorm.weight','output.LayerNorm.bias']\ndef expose(l, save_edit, base, set_structure):\n    P=f'bert.encoder.layer.{l}.'\n    save_edit(P+'attention.output.LayerNorm.weight', np.ones(768,np.float32))\n    save_edit(P+'attention.output.LayerNorm.bias', np.zeros(768,np.float32))\n    save_edit(P+'output.dense.weight', np.zeros((768,3072),np.float32))\n    save_edit(P+'output.dense.bias', np.zeros(768,np.float32))\n    save_edit(P+'output.LayerNorm.weight', np.ones(768,np.float32))\n    save_edit(P+'output.LayerNorm.bias', np.zeros(768,np.float32))\n    return set_structure(layers=list(range(l+1)), ffn_keep=3072)\ndef unexpose(l, drop_edit):\n    for n in EXPO_NAMES: drop_edit(f'bert.encoder.layer.{l}.'+n)\n'''\nopen('lib2.py','w').write(code)\nexec(code)\nprint(edits())\nprint(expose(0, save_edit, base, set_structure))\n"
}
```


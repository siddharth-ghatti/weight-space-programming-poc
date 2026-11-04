## reasoning summary

The best result so far is the base9 setup at 97.46%, ahead of base11 at 96.9%, and the tap doesn't help, so I'll drop the layer-6 tap and move the readout to layer 9. Layers 10 and 11 will be identity, and I'm setting up the layer-9 readout using the base C and M vectors from that layer.

I'm dropping the layer-6 tap edit and moving the readout to layer 9, generalizing the readout function to take a layer index. Layers 10 and 11 get identity settings with zeroed attention and output dense weights and biases and LayerNorm weight 1, bias 0, and I'm removing the old layer-11 readout edits, though the layer-11 query and value edits would be irrelevant either way.

I'm checking the identity layers now: layer 10 passes the CLS state through unchanged after LayerNorm, since the output dense is zeroed, so the CLS vector stays x9 at the end. The readout then becomes a linear function of the kept CLS channels and the mean hidden state, and I'll fit it with wc=1.0 and lam=500 on all 1496 examples. I still need to decide the channel dimensions and the scale factors, and then confirm with a predict pass that the offline logits match.



Repeated CV confirms the layer-9 readout is best at 97.46%, so I'm dropping the layer-6 tap and making layers 10–11 identity maps.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nsrc=open('readout.py').read()\nsrc=src.replace(\"def write_layer11_readout(R, save_edit, c=100.0, s=1.0, hch_mean=None):\\n    L='bert.encoder.layer.11.'\",\n \"def write_layer11_readout(R, save_edit, c=100.0, s=1.0, hch_mean=None, layer=11):\\n    L=f'bert.encoder.layer.{layer}.'\")\nsrc+='''\ndef write_identity_layer(save_edit, layer):\n    L=f'bert.encoder.layer.{layer}.'\n    z=np.zeros(768,np.float32); o=np.ones(768,np.float32)\n    save_edit(L+'attention.output.dense.weight', np.zeros((768,768),np.float32)); save_edit(L+'attention.output.dense.bias', z)\n    save_edit(L+'attention.output.LayerNorm.weight', o); save_edit(L+'attention.output.LayerNorm.bias', z)\n    save_edit(L+'output.dense.weight', np.zeros((768,3072),np.float32)); save_edit(L+'output.dense.bias', z)\n    save_edit(L+'output.LayerNorm.weight', o); save_edit(L+'output.LayerNorm.bias', z)\n'''\nopen('readout.py','w').write(src)\nimport readout\nfor e in list(edits()): drop_edit(e)\nF=np.load('feats.npz'); y=F['y']\nR=readout.fit_readout(F['C9'],F['M9'],y,wc=1.0,lam=500.0)\nlg=readout.logits_offline(R,F['C9'],F['M9']); print('train acc',(lg.argmax(1)==y).mean(), 'logit range',lg.min().round(2),lg.max().round(2))\nhm=F['C9'][:,R['ch']].mean(0); print('ch',R['ch'],'hm',hm.round(3),'std',F['C9'][:,R['ch']].std(0).round(3))\nreadout.write_layer11_readout(R, save_edit, c=100.0, s=1.0, hch_mean=hm, layer=9)\nreadout.write_identity_layer(save_edit,10); readout.write_identity_layer(save_edit,11)\nnp.savez('readout_L9.npz', **R)\nprint(sorted(edits()))\nopen('notes.md','a').write(\"\"\"\n- Repeated CV (5 seeds x5 folds): base readout at L9 [C9,M9] wc1 lam500 = 97.46% (best); L10 97.1; L11 96.9; layer6 tap adds nothing.\n- NEW SAVED EDITS: readout at layer 9 (readout_L9.npz, c=100,s=1), layers 10,11 identity; tap dropped. readout.py has fit_readout / write_layer11_readout(layer=) / write_identity_layer.\n\"\"\")"
}
```


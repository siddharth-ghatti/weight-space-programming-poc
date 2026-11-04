## reasoning summary

I'm checking the layer 11 edits and the pool design file to figure out how the install was done, since there's no install code in the libs. The attention output weights and bias show changes across all 768 entries, which might be a scaled identity or a shift, and the key weights have 589 changed values.

Next I'm checking the value and output projections in layer 11, since value looks unchanged while the attention output weights changed on 768 entries, which could mean a diagonal scaling. I'm weighing whether the key weights were zeroed out, which would make attention uniform, and whether the output bias follows from the diagonal.

I'm checking the layer 11 edits in the pool design file, since the install code isn't in the libs. The output dense weights for the eight RES rows and the pooler weights and bias also changed, so I'll look at pool_design.npz for the anchors and coefficients that would explain them.



The install was done inline earlier and isn't saved as a function, so I'm inspecting pool_design.npz alongside the current layer-11 and pooler tensors to reconstruct it.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom poolfit import *\nd=np.load('pool_design.npz'); print({k:(d[k].shape, d[k].dtype) for k in d.files})\nfor k in d.files:\n    if d[k].size<20: print(k, d[k])\nP='bert.encoder.layer.11.'\nfor nm in ['attention.self.query.weight','attention.self.query.bias','attention.self.key.weight','attention.self.key.bias','attention.self.value.weight','attention.self.value.bias','attention.output.dense.weight','attention.output.dense.bias','attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','intermediate.dense.weight','intermediate.dense.bias','output.dense.weight','output.dense.bias','output.LayerNorm.weight','output.LayerNorm.bias']:\n    a=W(P+nm); \n    if a.ndim==2:\n        off=a-np.diag(np.diag(a)) if a.shape[0]==a.shape[1] else a\n        print(nm, a.shape, 'norm',np.linalg.norm(a).round(3),'diag mean/std', (np.diag(a).mean().round(4), np.diag(a).std().round(4)) if a.shape[0]==a.shape[1] else '', 'offdiag norm', np.linalg.norm(off).round(3))\n    else:\n        print(nm, a.shape, 'mean',a.mean().round(4),'std',a.std().round(4), 'RES', a[RES].round(3))\npw=W('bert.pooler.dense.weight'); pb=W('bert.pooler.dense.bias'); cw=W('classifier.weight'); cb=W('classifier.bias')\nprint('pooler rows0-7 nnz cols', [(np.abs(pw[i])>1e-8).sum() for i in range(8)], 'pb', pb[:8].round(3), 'rest norm', np.linalg.norm(pw[8:]))\nprint('classifier', np.round(cw[:, :10],3), cb)"
}
```


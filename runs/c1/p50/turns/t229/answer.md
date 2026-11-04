## reasoning summary

I'm double-checking that the output bias I flagged earlier is consistent: the zero absmax for the b2 bias doesn't contradict the earlier mean and std, since those were computed over all 768 entries, including the residual ones, of which only eight are nonzero. Next I'm setting up a numpy calculation of the predicted logits from the predicted features, to compare against the actual output after the pass.

Next I'm laying out the verification pass: reading back the edited model on the 1024 test ids at layer 7 with the logits output, then comparing those logits against the predicted ones from the predicted features, against the predicted ones from the actual layer-7 features, and checking the new carry and training accuracy.



Installation is complete with the parameter count still at 55,954,696; I'm computing the numpy-predicted logits now so the verification pass can compare against them, and it will also recheck the pool and the new carry.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\ncode=r'''\nimport numpy as np\nfrom poolfit import *\ndef pool_logits(M, C, W):\n    P='bert.encoder.layer.11.'\n    Wo=W(P+'attention.output.dense.weight').astype(np.float64); bo=W(P+'attention.output.dense.bias').astype(np.float64)\n    Wv=W(P+'attention.self.value.weight').astype(np.float64); bv=W(P+'attention.self.value.bias').astype(np.float64)\n    g1=W(P+'attention.output.LayerNorm.weight').astype(np.float64); be1=W(P+'attention.output.LayerNorm.bias').astype(np.float64)\n    W1=W(P+'intermediate.dense.weight').astype(np.float64)[:1024]; b1=W(P+'intermediate.dense.bias').astype(np.float64)[:1024]\n    W2=W(P+'output.dense.weight').astype(np.float64)[:, :1024]; b2=W(P+'output.dense.bias').astype(np.float64)\n    g2=W(P+'output.LayerNorm.weight').astype(np.float64); be2=W(P+'output.LayerNorm.bias').astype(np.float64)\n    pw=W('bert.pooler.dense.weight').astype(np.float64); pb=W('bert.pooler.dense.bias').astype(np.float64)\n    cw=W('classifier.weight').astype(np.float64); cb=W('classifier.bias').astype(np.float64)\n    ctx=M@Wv.T+bv\n    z=LN(ctx@Wo.T+bo+C)*g1+be1\n    h=LN(z+gelu(z@W1.T+b1)@W2.T+b2)*g2+be2\n    return np.tanh(h@pw.T+pb)@cw.T+cb\n'''\nopen('lib10.py','w').write(code)\nimport numpy as np, json\nfrom lib10 import *\nfrom lib7 import labels_for\nd=np.load('final_design.npz'); Mn,Cn,Sdep=d['Mn'],d['Cn'],d['Sdep']\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS)\nlg=pool_logits(Mn,Cn,W)\nprint('pred acc', (lg.argmax(1)==y).mean(), 'agree with Sdep', (lg.argmax(1)==Sdep.argmax(1)).mean())\nprint('logit range', lg.min().round(3), lg.max().round(3), 'max |tanh arg| proxy', np.abs(np.arctanh(np.clip(lg,-0.999999,0.999999))).max().round(3))\nsrt=np.sort(lg,1); print('min margin', (srt[:,-1]-srt[:,-2]).min().round(4))\n# check proportionality of logit-arg with Sdep centered\nA=np.arctanh(lg); Ac=A-A.mean(1,keepdims=True); Sc=Sdep-Sdep.mean(1,keepdims=True)\nk=(Ac*Sc).sum(1)/(Sc**2).sum(1); print('per-sample scale mean/cv', k.mean().round(4), (k.std()/k.mean()).round(4), 'resid', np.abs(Ac-k[:,None]*Sc).max())\nnp.save('pred_logits_final.npy', lg)"
}
```


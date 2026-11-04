## reasoning summary

I'm working out how much variance pruning the bottom 512 neurons removes per layer. It comes to roughly 8-10%, which implies about a 30% relative change in the FFN output, matching the drift I saw. Pruning only 256 should remove about half as much, so I'm estimating a leave-one-out drop of around 0.2-0.25 for m=2816 (

I'm weighing a safer choice of m=3072 at 95.3M against m=2816 at 91.4M, which costs about 0.2 in the leave-one-out drop. The flat importance suggests this FFN won't prune well without retraining, and the 512-neuron drift figures still look real despite the noisy 0.48 subset estimate. Since the size gain is only about 4%, while the failure risk may rise from roughly 8% to 12-15%, I'm leaning toward the larger, safer model.

I'm checking whether the m=3072 design really generalizes as well as the leave-one-out numbers suggest, since the KRR head was tuned on the same data. I'm also looking at whether dropping just one layer might hurt less than dropping two, given that the earlier two-layer drop lost more than FFN pruning did.

I'm planning the remaining experiments now, with m=2816 as the candidate: first a structure pass reading 1024 ids on layers 9 and 10 to get a leave-one-out comparison against the baseline, then a second read of the remaining 472 ids, followed by a verification pass and a fallback check if the drop comes out too large.

I'm weighing a fallback to m=3072, which would need reads of all 1496 ids to fit the head, plus a verification pass, leaving no spare pass. Alternatively, I could skip new reads for m=3072 by reusing base-config layer-9 outputs, but that only works if layers 0-8 are unchanged, so I'd first have to drop the output bias compensation I added to those layers.

I'm checking how the pooling layer behaves: with the query at zero, attention is uniform over the real tokens, so the pooled vector is just their mean. That means the residual dimensions come out zero, and the question is whether the FFN is actually zeroed, since I haven't changed the output projection weights.

Maybe the intermediate weights are zeroed with a large negative bias so GELU outputs roughly zero, leaving only the output bias b2 from layer 11. I'm checking whether that b2 is already zero from the original edit, and if the final CLS can be computed from the base reads.

I'm checking whether I can skip the extra passes: if the numpy formula for the final CLS, built from base reads, matches the pass_5 data (cls_hidden_L10 against the layer-8 outputs), then m=3072 needs no new reads. For m=2816 I'd still need edited reads of the layer-8 outputs.

I'm going to check the closed-form formula against the pass_5 data now, so I'll compare my numpy-computed final CLS with the cls_hidden_L10 values I already have.



Importance is flat across FFN units, so a 2816 prune would probably cost around 0.2. I'll test whether LN(LN(cls + 5·mean)) reproduces the final CLS from pass_5, so I can get features for unpruned configurations without more passes.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nfor nm in ['attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','output.LayerNorm.weight','output.LayerNorm.bias','output.dense.bias','intermediate.dense.bias','attention.self.value.bias','attention.output.dense.bias']:\n    a=W('bert.encoder.layer.11.'+nm); print(nm, a[:4], a[RES][:3], np.abs(a).max())\nWi=W('bert.encoder.layer.11.intermediate.dense.weight'); print('Wi absmax', np.abs(Wi).max())\nO=W('bert.encoder.layer.11.attention.output.dense.weight'); V=W('bert.encoder.layer.11.attention.self.value.weight')\nprint(np.allclose(V,np.eye(768)), O[0,0], O[RES[0],RES[0]], np.abs(O-5*np.diag(np.diag(O)/5)).max())\nM=np.load('pass_5/mean_hidden_L9.npy'); C=np.load('pass_5/cls_hidden_L9.npy'); C10=np.load('pass_5/cls_hidden_L10.npy')\ndef ln(X,g,b,eps=1e-12):\n    mu=X.mean(1,keepdims=True); v=((X-mu)**2).mean(1,keepdims=True); return (X-mu)/np.sqrt(v+eps)*g+b\ng1=W('bert.encoder.layer.11.attention.output.LayerNorm.weight'); b1=W('bert.encoder.layer.11.attention.output.LayerNorm.bias')\ng2=W('bert.encoder.layer.11.output.LayerNorm.weight'); b2=W('bert.encoder.layer.11.output.LayerNorm.bias')\nob=W('bert.encoder.layer.11.output.dense.bias')\nx1=ln(C + M@O.T, g1, b1); out=ln(x1+ob, g2, b2)\nprint('max err', np.abs(out-C10).max(), 'rel', np.linalg.norm(out-C10)/np.linalg.norm(C10))\n"
}
```


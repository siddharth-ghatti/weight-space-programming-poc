## reasoning summary

Before committing a pass to Exp E, I'm weighing whether it's the most useful experiment. Exp C already suggested quality tracks layer count, and it covered a 7-layer edited stack against base L7. So dropping layer 4 alone probably lands near the same result.

I'm questioning whether dropping layer 4 alone is worth a pass, since the earlier result on a 7-layer edited stack already sat near the base L7 quality. Dropping layers seems to track layer count rather than specific layers, so Exp E probably teaches little new. Still, the open question is whether some particular layer can be removed for free, and I'm also wondering if the final BERT layer could double as the pooling step.

I'm weighing whether a pooling layer could double as the final BERT layer, since only its output at CLS matters. The 10-layer setup would then reuse layer 9 to pool layer 8's output into the mean features. I'm also considering whether its attention heads and FFN could add something beyond a plain mean.

I'm checking whether the start model's carry heads already give me the layer-2 and layer-6 means, since L2 and L6 each write ridge scores of their inputs into the reserved dims. If so, I could feed M2 and M6 in alongside the pooled M9 without adding another layer.

I'm working out how to get the M2 and M6 means into the reserved dims through the existing carry heads, by refitting the L2 head's value and output weights. Then the pooling slot at CLS would take the mean of M9 plus those reserved-dim scores, and I'm considering a feed-forward nonlinearity there, possibly with random features, to approximate a kernel on M9.

I'm wondering whether computing GELU features from the read activations would count as simulating part of the forward pass, which the rules forbid. Designing a kernel and fitting a readout on it looks like the earlier ridge work, but running an FFN unit's exact computation on the pooled input is closer to reimplementing the model. I'm not sure yet where that line falls.

I'm considering a two-stage setup for the FFN: set its first layer to produce 3072 features, then read the 768-dimensional output after LayerNorm and fit ridge on that, which keeps the step to reading activations and fitting in closed form.

Kernel features on M9 seem worth roughly half a point over the linear fit, which matters for the final call. I'm also checking whether the test score of 98.31 holds up against that CV figure before deciding how much to trust either.

Checking whether a readout built from kernel features on the pooled mean beats the linear fit before I settle on a layer count. I'm thinking about the pooling slot's FFN as the place to get that nonlinearity, but I still need to recompute the CV numbers for kernel(M_j) plus linear M2 at j = 6 through 9.

I'm working through how the pooling slot's attention would be set up: with zero query weights and bias, all keys score equally, so attention is uniform over real tokens. The head values would then produce a linear projection of the mean, but the CLS token's own residual can't be subtracted out, since attention is uniform and can't isolate it.

I'm working out the pooling slot's feed-forward layer: each of its units would apply a GELU to a projection of the normalized CLS vector, which would give a kernel-like readout on the mean. Checking whether that nonlinearity is worth the setup, I'm weighing it against a plain linear fit.

I'm checking whether the output LayerNorm would wash out the small class-score contributions from the FFN, since the residual's scale would vary from sentence to sentence. Fixing the attention LayerNorm gain so the residual has a constant std might keep those contributions stable.

I'm checking whether the output LayerNorm would wash out the small FFN class-score contributions, since the residual's scale might drift between sentences. If I fix the attention LayerNorm gain to a constant, the residual's std stays put, and the reserved dimensions would contribute linearly. The worry is that ridge on LayerNorm-normalized means may not match the z-scored CV numbers I computed earlier.

Trimming the 3072-feature FFN idea down: the start model already carries the layer-2 and layer-6 signals, so the simplest route is to keep that setup and remove layers rather than build new machinery.

I'm planning to keep layers 0 through 8 plus the final pooling layer, which gives a 10-layer model. The layer-10 carry head would be lost, but the layer-2 and layer-6 carry heads stay in place. I'd then refit the pooler on the final CLS hidden state and check whether that linear readout still lands around 97.5 to 97.8 in CV.

I'm planning to carry the reserved-dimension scores forward honestly, so the CV doesn't leak. The plan is to fit carry scores on the layer-2 and layer-6 raw means inside each fold, then write those weights in from the full-data fit.

Now I'm weighing the final architecture against the quality target by checking candidate linear readouts on the means from layers 2, 6, and 8 or 9, including the pooling-layer variant. I'll run a quick multi-seed cross-validation on each to see which is worth implementing.



Exp C suggests dropping a layer costs about one layer's worth of quality, so I'm re-running the key numbers from saved base reads with honest 5-fold CV before deciding whether Exp E is worth a pass.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import *\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy'); print('reserved',R)\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\nC={j:d[f'cls_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\nfor combo in [(9,),(2,9),(2,6,9),(1,5,9),(2,6,8),(2,8),(8,),(2,5,8),(2,6,7),(2,7),(2,6,10),(2,6,9,10)]:\n    X=np.hstack([M[j] for j in combo])\n    r=cv_acc(X,y,lams=(100,300,1000,3000),seeds=(0,1,2))\n    print(combo, {k:round(v*100,2) for k,v in r.items()})"
}
```


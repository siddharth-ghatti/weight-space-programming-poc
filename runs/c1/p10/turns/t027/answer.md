## reasoning summary

I'm checking how much each layer shifts the centered representation, and whether the downstream layers would absorb a dropped layer. I'm also weighing whether a different approach would work better, since the kernel and the 9-layer model differ by only about a point of leave-one-out accuracy. Then I'll reconsider the question from the pass-chance angle, and I'm still working out the threshold.

I'm weighing the 97.8% threshold against the leave-one-out estimates: the two-layer option sits at 97.9% and the nine-layer model at 98.6%. Given an unknown test size, the failure odds swing from about 4% to over 40% depending on the true accuracy, so the gap between the models matters. I'm now looking for a depth that lands near 98.3%, and checking whether any of the misclassified items could be recovered with feature engineering.

I'm working through a trick to get the mean activations without a full forward pass: adding a large constant vector to the FFN output bias so that the layer norm's mean and variance get dominated by that constant, which might make the per-token nonlinearity behave linearly.

I'm looking at whether the mean activations can be pulled out of a single pass by pushing a large constant into the layer norm, so the output becomes linear in the FFN gate values. The trick seems to work numerically with enough signal-to-noise margin, but the downstream layers would then receive a distorted input.

I'm looking at whether the per-unit mean activations can come from a single probe pass: a large constant in the output bias would make the layer norm roughly linear in the FFN values, so the token means could be read off. The catch is that the 768-dim projection only exposes part of the 3072 units, and the residual stream still mixes in, so I'd need extra probes to separate the pieces.

I'm weighing whether pruning FFN units without data is worth the cost: probing each layer would take 36 passes, and the savings look modest, about 8M parameters for k=10 with 512 units dropped per layer. Compared with dropping whole layers, that gain is small enough that the gamble may not pay off.

I'm weighing whether dropping 1024 units per layer would hurt much: the bottom half of each layer's units seems to carry roughly 12 to 25% of the variance, and that loss could compound across nine layers. Pruning this way looks likely to cause significant drift, so I'm reconsidering the trade-off.

I'm weighing the three realistic options: the full 10-layer model at 95.3M parameters, a pruned 8 or 9 layer version with uncertain accuracy, and the shallow 2-layer model at 38.6M, which reaches about 97.9% LOO but carries more risk. The shallow option only loses about 10 of 1496 sentences compared with the deep model, so I'm asking whether a 3-layer variant could close that gap.

I'm checking whether the CLS token's own state could be shaped through layer 0's attention, so the final vector carries more than a linear projection of the mean of layer 1's states.

I'm considering whether the CLS token's own state could carry a nonlinear function of the layer-0 output, by having a head in layer 0 attend in a position-specific way. Since the CLS embedding is constant, its query is fixed, so such a head might shape the CLS state without disturbing the other tokens much.

I'm weighing whether a contextual approach would be needed, since reserving dimensions for the head output gets complicated and the kernel on the layer-0 features already reaches about 97.5%. So I'll accept that depth is necessary and focus on which layers can be dropped with the least loss for the classifier.

I'm considering whether to drop a few middle layers, since removing one in the 3 to 9 range should hurt MLM less than removing the first or last layer. Then I'm thinking about which layers matter most for classification, checking how the pooled features shift from one layer to the next and where the leave-one-out accuracy jumps, starting with layer 8, which lifts it from 98.0% to 98.6%.

I'm planning a budget of about seven passes, with four for exploration. The next experiment would install the pooling layer on top of layers 0 through 8 and read out the mean and CLS features in one go.

I'm planning the next experiment as a layer-dropped candidate, something like dropping layers 7 and 8, and reading out 1024 samples to compare against the full nine-layer baseline on the same subset. I'm also setting aside the idea of changing the FFN structure, since the stock BERT weights don't support that.

I'm checking whether the shallow model is really that much weaker, so I'm calibrating the leave-one-out score against the original head's setup: ridge on the three carried layer averages plus the final CLS, to see how well that predicts the test accuracy.

I'm checking the leave-one-out score of a ridge fit on the middle and final layer features, comparing it against the 98.31% test accuracy the original model reached, to see whether LOO is a reliable predictor here.



Every layer shifts the representation by about 45–50%, so none can be dropped safely. Before choosing how much risk to take, I'll check my accuracy estimate against the known leave-one-out score of the original head, which is 98.31%.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}; C={L: F[f'cls_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef loo_ridge(X, y, lams):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; T=np.eye(8)[y]-1/8\n    U,s,Vt=np.linalg.svd(Z,full_matrices=False); out={}\n    for l in lams:\n        f=s**2/(s**2+l); Hd=((U**2)*f[None]).sum(1); Yh=U@(f[:,None]*(U.T@T))\n        out[l]=round(float((((Yh-Hd[:,None]*T)/(1-Hd[:,None])).argmax(1)==y).mean()),4)\n    return out\nprint('orig-style ridge M2,M6,M10,C12', loo_ridge(np.concatenate([M[2],M[6],M[10],C[12]],1),y,(100,300,1000)))\nprint('5fold same', [round(cv_acc(np.concatenate([M[2],M[6],M[10],C[12]],1),y,l,seeds=(0,1,2)),4) for l in (100,300,1000)])\n# editor's own channel scores at final layer: logits are training fit; ignore\n"
}
```


## reasoning summary

I'm checking whether the mean|dm| of 0.0756 for m2048 is comparable to the 0.256 noted for m1536, since the margin-noise definition might differ between them. I'll compute both to see if they line up.

I'm weighing layer count against width for the size budget: six layers at width 1536 comes to about 68M, while the wider options either overshoot or cost a layer. Since the feed-forward part dominates each layer, narrowing it saves the most, and I'm still checking how many net test errors the 76.5% threshold would tolerate.

The test set is probably 400 items, so with a base accuracy of 77.5% and a 76.5% threshold I can afford to lose about four net correct answers. The in-sample flips mostly hurt because the base model is already right on 92.5% of those items. On test, the base is wrong more often, so flips could land either way, and the net loss should be small but noisy.

I'm estimating the test flip rate from the in-sample rate, though test items likely sit closer to the decision boundary, so flips could be more frequent there. With about 4 flips on 400 items and roughly half of them hurting, the expected net loss is only one or two correct answers, which seems acceptable. The governance class is small, so macro-F1 might move by around 0.02. The real question is how aggressive to be with the change, and whether I can estimate the test-side flip rate more directly from the base accuracy gap.

I'm weighing whether to keep the var-based ranking and ridge compensation, which rests on only about 5,000 tokens from 32 probe ids for a 2048x768 fit, versus fitting the compensation on corrected inputs so upstream errors are accounted for. I'm also considering refitting the classifier head in closed form on the edited model's last-layer features.

I'm considering whether to refit the classifier head in closed form, either just the output projection or an affine correction on the CLS features mapped back to the base model's, with ridge to limit overfitting on 1000 samples. Folding such a correction into the dense layer would keep the edit closed-form. I'm also weighing the size target against the layer and width options, since the 6-layer, width-1536 variant lands near 68M.

I'm weighing the remaining size options, where dropping a layer at smaller widths saves the most parameters, since the fixed attention cost per layer dominates there. A layer-four drop with a refit head looked like roughly 4.5% flips in-sample, which seems too many to accept.

I'm now weighing narrowing against dropping layers: at width 3072 the 5-layer option comes out worse than the 6-layer width-1536 one, so narrowing the feed-forward part looks like the cheaper route. Next I'm asking whether better compensation could come from more fitting data, since each probe pass yields only about 32 ids' worth of exact FFN inputs.

I'm working through the probe trick's arithmetic: with LN1 set to identity, the attention residual comes out un-normalized, so the layer's output is just that sum with the FFN contributing nothing. I'm checking how that relates to the LN(z) I need, and whether the gain and bias terms recover the correct input.

I'm working through whether the layer's output can be recovered from the probe: with the weights set to identity, the attention residual comes out unnormalized and the FFN contributes nothing. I'm checking how that residual relates to the LN(z) I need, and whether the gain and bias terms bring back the correct input. Since each probe pass only exposes one layer, I'm also weighing whether a numpy reconstruction of the FFN on probed inputs is acceptable for fitting the compensation.

I'm getting at the same compensation-fitting question from a different angle: the layer-5 case only needs the CLS position, since that is the only output that reaches the classifier head. So I have CLS-level FFN inputs for 1000 ids to fit against, which is much more data than the 5049 tokens from the 32-id probes.

I'm looking at whether, for layer 5, I can prune the FFN more aggressively, since only the CLS position feeds the head and I have 1000 exact CLS inputs to fit a compensation against, with cross-validation to guard against overfitting. I'm also considering whether to keep the units that best reconstruct the output, given that each unit is a gelu of a linear function of the input.

I'm noting that the layer-5 FFN has to match the same width as every other layer, so pruning it alone doesn't help. Since the total size is the number of kept layers times the per-layer cost, the hardest layer to narrow sets the bottleneck. I'm wondering whether dropping layer 5 and approximating it could work, though its attention gathers information for the CLS token, so that seems hard to do cheaply.

I'm weighing whether the layer-5 FFN can be cheaply narrowed, but every layer has to keep the same width, so the hardest layer sets the bottleneck. A second option is more fitting data for the compensation, since each probe pass only yields about 5,000 tokens for one layer, and I'm wondering whether a data-free or ridge-based fit could do better with more of them.

I'm checking whether a split-half test would show if more fitting tokens would actually help: if out-of-sample reconstruction error is close to in-sample, extra probe passes won't buy much. I'm also looking at the importance scoring, which currently uses variance times the squared norm of the W2 column; a selection that accounts for the compensation step, like a Gram-weighted subset selection, might pick better units.

I'm weighing a greedy removal scheme instead: the cost of dropping a unit, given the others are refit, looks like its W2 column norm divided by the corresponding diagonal entry of the inverse Gram matrix, and I'd remove the cheapest unit and update the inverse each step.

I'm planning a greedy removal of units, using a closed-form OBS-style update instead of gradient training. The cost is manageable in numpy, though the Gram matrix from only 5049 tokens will need dampening to avoid overfitting. I also want to check whether the downstream effect matters more than the raw FFN output error.

I'm laying out a plan: first check whether the holdout reconstruction error is close to the in-sample error, which tells me whether more fitting tokens would help. Then I'd compare an OBS-style greedy pruning against the variance-ranked selection at m=1024 and m=1536, and finally re-probe each layer in the already-edited model so its compensation fits the base targets.

I'm now sketching the sequential correction step: after pruning earlier layers, I'd re-probe each layer on the edited model and fit its compensation so the pruned FFN plus the edited residual matches the base layer's pre-LN2 output. That needs token-aligned probes for the same ids, which I already have for the 32 ids, at a cost of about one probe pass per layer. I'm also checking whether the error after LN2 can be treated as roughly matching the pre-LN outputs.

I'm weighing whether the target of 6 layers at width 1024 or 5 at width 1536 is realistic, since each step down saves roughly 5M parameters but failing the accuracy bar would be far worse. I'd want agreement around .975 in-sample and at most about a 1% accuracy drop, and the in-sample drop at m1536 is already around 2 points, though test flips may hurt less.

I'm wondering whether the test flip rate could be estimated better than from the in-sample numbers, since the base margins look inflated. Test likely has more low-margin items, so 23 in-sample flips could mean roughly 12-16 on 400, which would be borderline against the 4-item budget. I'm considering whether to reduce the noise further through better fidelity before going smaller.

I'm thinking about checking the flip rate by taking the margin distribution from a CV probe on labels as a stand-in for test margins, then resampling the edited model's logit noise onto it to estimate test accuracy change.

Next I'll check the size of the FFN files so I can confirm the data is there before running the data sufficiency check and the OBS pruning.



Pass 12 matches the base model on 98.4% of items, with in-sample accuracy .917 against .925. Next I'll check whether the 5,049 probe tokens are enough for the FFN compensation, then compare var-rank+ridge with an OBS-style greedy selection.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nexec(open('builder.py').read())\npid = json.load(open('probe_ids.json')); print(len(pid), pid[:5])\nfor L in range(5):\n    A = np.load(f'ffn_in_L{L}.npy'); print(L, A.shape)\nA5 = np.load('ffn_in_L5_cls.npy'); print('L5', A5.shape)\niscls = np.load('ffn_in_L0_iscls.npy'); print(iscls.shape, iscls.sum())"
}
```


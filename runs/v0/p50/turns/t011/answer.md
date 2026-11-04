## reasoning summary

Trying to keep all 768 residual dimensions free for a pooled subspace would damage BERT, so I'm looking at compressing instead. The ridge probe only needs 8 outputs per layer, so the earlier layers' means could be reduced to a small set of features and carried forward at CLS.

Working through the CLS vector: I could project the pooled layer's logits onto a few directions and write them into the residual stream, but the later layers would process and wash them out. Blending in a large mean term damages the signal too, so the 8-dimensional logits are what really matter. I'm weighing whether a low-rank output map in layer 11 could place those logits in specific residual directions while the CLS stream carries earlier-layer information alongside.

The attention output at CLS could be a rank-8 map pushing the logits into chosen residual directions, so the scale of the normalized vector shouldn't flip any argmax. The remaining worry is the mean subtraction and the other components projecting onto the readout directions. That noise can be absorbed into the ridge fit, since the fit runs on the actual final vectors anyway. So what matters is simply which features the final CLS vector ends up containing.

Making only the CLS token attend uniformly at an early layer is hard, since changing the query weights would alter every token's attention. I'm considering whether a constant key component could shift the scores so CLS's attention flattens without touching the other positions.

I'm weighing whether a head in layer L could be repurposed for uniform attention, but zeroing its query would hit every token, so each would receive the same sentence-mean vector and the perturbation might be too large. The gain looks like roughly 1% on CV, within about ±0.4% noise, so it may be worth it but the setup is complicated. I'll build and verify the single-layer version first, then revisit.

Now I'm weighing whether the final CLS vector could blend the layer-11 mean with the layer-11 CLS state, normalized, and whether a CV check on concatenated mean and CLS features would show any gain.

Next I'll check whether a plain concatenation of the layer-11 mean and CLS states gives a better CV score. The sum version can't do that, because a single readout weight vector would be applied to both parts. Unless the mean maps into a subspace where CLS barely varies, which doesn't look true since CLS varies in many dimensions. I'll implement the single-layer version first and decide after seeing the numbers.

Setting the layer-11 FFN to zero and the LayerNorm gains to one, so the final CLS becomes a normalized sum of the CLS residual and a scaled mean. I still need to pick the scale c, which depends on how the norms of the mean and CLS vectors compare.

Setting c to 1000 should make the CLS contribution negligible, though I'd still like to check whether a small CLS component helps, by scanning normalize(c·mean11 + cls11) across values of c. Once the feature is built, I'll fit a ridge readout on the final normalized CLS vector and wire it into the pooler and classifier.

Checking the sign of the temperature scaling: tanh is monotonic, so each logit's ranking across classes is preserved and the argmax stays the same. Saturation can only create ties, so I'll keep the scale small enough, something like 0.1, to stay in the linear range. Then a quick forward pass should confirm whether the features really come out as the normalized layer-11 mean.

I'm going to fit the ridge on the normalized layer-11 mean, since that avoids extra forward passes. Fitting and evaluating on the same practice set would only give training accuracy, though, and the cross-validation estimate of about 97% is the better guide. Before relying on the mean, I want to check its norms and whether it includes the CLS and SEP positions, which I can do by comparing edited-model reads against the base-model mean.

I'm checking that the edited model's CLS output matches the normalized layer-11 mean from the base model, which will confirm whether the attention mean is the same as the mean-hidden feature. Then I'll write the edits.



Combining layers might add about 1% (98.1% CV), but that's harder to realize, so I'm first building a clean single-layer version. Next I'll check vector norms to pick the scaling constant c.



## tool_use: run_python

```json
{
 "code": "M11=np.load('feats/mean_L11.npy'); C11=np.load('feats/cls_L11.npy')\nprint('mean11 norm', np.linalg.norm(M11,axis=1).mean(), 'cls11 norm', np.linalg.norm(C11,axis=1).mean())\nprint('mean11 centered norm', np.linalg.norm(M11-M11.mean(1,keepdims=True),axis=1).mean())\nprint(np.abs(M11).max(0).argsort()[-5:], np.abs(M11).max())\nprint(base('bert.encoder.layer.11.attention.output.LayerNorm.weight')[:5])\n"
}
```


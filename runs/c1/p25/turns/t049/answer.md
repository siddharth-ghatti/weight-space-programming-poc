## reasoning summary

I'm reading the starting readout CV at about 97.8, averaging 33 errors across seeds. The three candidates average 32.7, 30, and 29.3 errors, and each pair disagrees on roughly 10 items per 1496, so on a test set they differ from the start on about 1.4% of items each way. I'm now heading toward the M8+lin(2,6) candidate.

I'm weighing the two candidates on test risk: with roughly 600 test items, the net difference from the starting model is only a few items, so the failure chance for the (8,(2,6)) option comes out around 12%. The (9,(2,6)) option looks slightly better on net expected errors, though the (8,(2,6)) pick earned its place through a broader search and is still a fair contender.

I'm weighing whether the k=9 model fits the existing setup: layers 0 through 7 stay as they are, the L10 carry head drops out, and the pool slot at L11 should still be able to read the reserved dims that carry the scores into the final CLS. Since the k=9 variant is about 88M parameters against roughly 95M for k=10, the smaller size is also worth considering.

I'm weighing whether a k=9 model could use layers 0 through 8 with the last layer doing double duty, but the CLS output of layer 8 only sees layer 8's inputs through its attention, so it can't pool the full token mean the way the separate pool slot does.

I'm weighing whether the last layer's pool-slot FFN, with its 3072 hidden units at CLS, could serve as a richer nonlinear readout on the pooled M8 features, the way the quadratic kernel already approximates. The quadratic kernel beat RBF and added about 0.3 over linear, so I'm asking whether a learned attention pooling over tokens could extract more than the uniform mean.

I'm weighing whether to build the pool-slot readout for k=10 first, since it gives more margin, and then check whether k=9 holds up under the same cross-validation before choosing between them.

I'm checking the start model's last pooling layer: its attention is set to uniform, so the [CLS] token gets roughly twice the mean of the token states plus its own state. Its output projection is zeroed, so the layer's contribution is just that normalized pooled value, and the final CLS vector comes from two layer norms around that.

I'm considering whether the pool slot's FFN, with its 3072 GELU units, can serve as the readout instead of the pooler, fitting the output weights by ridge regression on those features. The open question is how to get the LN1 output at CLS, since I can't read it directly.

I'm checking whether the last layer's FFN can be read out directly: if I zero its output and set the final norm to identity, the CLS state would just be the normalized input, but computing the FFN features myself in numpy starts to look like reimplementing part of the forward pass.

I'm checking whether the FFN's effect can be read through the final CLS state: if its output projects the 3072 features into 768 dims and LN2 follows, then reading h gives a 768-dimensional view of the readout without computing the FFN myself.

I'm weighing a design where the pooler's 768 tanh features, computed on the normalized M8 representation plus the carried scores, feed a ridge classifier. I want to check whether 768 of those features can match the quadratic kernel's cross-validated performance, using numpy only for the design estimate and the real pooled outputs for the final fit.

I'm weighing whether the pooler's 768 tanh units can approximate the linear and quadratic kernel terms over the practice centers, since tanh with a nonzero offset picks up a quadratic component. I'd also need a linear pass-through, which would take some of those units. Separately, the M2 and M6 carried scores were fit in the joint kernel, so for an honest estimate I'd refit the carry heads' value and output weights to write their own linear parts.

I'm weighing whether to keep the start's carry heads and treat the eight reserved dims as fixed features, but that would leak the joint fit into cross-validation, so I'd need to simulate those carried scores within each fold to get an honest estimate.

I'm trying to see how the carry heads in layer 2 are built: the query is zeroed, and the value and output weights are changed on a large number of entries, which suggests they route only a handful of dimensions into the reserved slots.

I'm checking the carry head's arithmetic: its value and output weights cover 64 rows across all 768 columns, plus the reserved columns, which accounts for the 54,784 changed entries. So each head reads its mean-pooled value and writes a fixed offset into the reserved dims, and the attention layer's output norm then operates on that combined vector.

I'm following how the carried scores end up at the final CLS: each reserved value gets rescaled by the per-token layer norm and offset by its shared mean, so the decode has to subtract the mean across the eight reserved dims.

I'm considering whether the carried linear scores should be built in directly, by setting the carry heads' value and output weights to my own projection, instead of leaning on the start's fit. The alternative is to drop the linear carry entirely and check whether the quadratic kernel on the normalized M8 features already does most of the work without it.

I'm thinking about whether the carried scores should be built in as the reserved dims' sum of the two linear terms, since the classifier just adds them to the kernel part. That would let the joint fit set those weights directly, and I'd only need to check how the final logits depend on the pooled output.

I'm going to check the start model's L11 and L2 carry weights to see how stable the carry scale is across sentences, since exact additivity depends on it. If it's roughly constant, I can use a minimal structure that keeps all the existing edits, read the final CLS and pooled outputs for all 1496 items, and then refit only the pooler and classifier.

I'm concerned that the reserved dims carry start's fitted M2 and M6 scores, which were trained on all 1496 items, so in-sample they look better than they will on held-out data. If I refit the classifier on those scores as they stand, it would overweight them, and the cross-validation numbers would be optimistic.

I'm weighing whether to rebuild the carry heads so the linear scores are fixed by design rather than inherited from the start's fit. Since the classifier simply adds those scores to the kernel part, I'd fit the joint model on the actual pooled features plus the linear terms, keeping the combination weights fixed instead of tuning them against leaked values.

I'm settling the readout design: the kernel features will come from pooler rows that ignore the reserved dims, and a separate set of pooler rows will decode the carried scores linearly. That way fitting the carry weights won't change the kernel features.

I'm checking whether κ, the per-sentence carry scale from the layer norms, stays close to constant, since that decides whether the carried scores can be treated as exactly additive. If it varies noticeably, I'd measure it per sentence with a probe and scale the regression features by it, the way the earlier fit did.

I'm checking how the reserved dims look at the CLS and mean-pooled reads, and whether their scale shifts from sentence to sentence. If it does, I'll measure it per sentence and scale the carried-score regressors accordingly.

I'm checking whether the reserved dims' scale stays steady from sentence to sentence, starting with the start model's layer-2 carry head to see how it routes the scores.



Depth-8 is roughly as accurate as the start readout (about 32.7 vs 33 CV errors out of 1496), and depth-9 is slightly better. Next I'll look at the L2/L6 carry heads and the L11 pooling layer to see how to build either one.



## tool_use: run_python

```json
{
 "code": "R=np.load('reserved.npy')\nfor L in (2,6):\n    p=f'bert.encoder.layer.{L}.attention.'\n    q=base(p+'self.query.weight'); bq=base(p+'self.query.bias')\n    v=base(p+'self.value.weight'); bv=base(p+'self.value.bias')\n    o=base(p+'output.dense.weight'); bo=base(p+'output.dense.bias')\n    qn=[np.abs(q[h*64:(h+1)*64]).sum() for h in range(12)]\n    h=int(np.argmin(qn)); print('L',L,'carry head',h, 'q norms',np.round(qn,1)[h], 'bq', np.abs(bq[h*64:(h+1)*64]).sum())\n    vh=v[h*64:(h+1)*64]; print(' value rows nonzero', (np.abs(vh).sum(1)>0).sum(), 'bv nz', (bv[h*64:(h+1)*64]!=0).sum())\n    oh=o[:,h*64:(h+1)*64]; print(' out nonzero rows', np.where(np.abs(oh).sum(1)>0)[0], 'cols nz', np.where(np.abs(oh).sum(0)>0)[0])\n    print(' out block', np.round(oh[R][:, np.abs(oh).sum(0)>0],3))\n    print(' bo[R]', np.round(bo[R],4))\n    g=base(p+'output.LayerNorm.weight'); b=base(p+'output.LayerNorm.bias'); print(' attnLN g[R]',np.round(g[R],4),'b[R]',np.round(b[R],4))\n    g=base(f'bert.encoder.layer.{L}.output.LayerNorm.weight'); b=base(f'bert.encoder.layer.{L}.output.LayerNorm.bias'); print(' outLN g[R]',np.round(g[R],4),'b[R]',np.round(b[R],4))\nfor L in (3,7,8):\n    g=base(f'bert.encoder.layer.{L}.output.LayerNorm.weight'); ga=base(f'bert.encoder.layer.{L}.attention.output.LayerNorm.weight'); print(L,'LN g[R]',np.round(ga[R],3),np.round(g[R],3))\nprint('L1 outLN', np.round(base('bert.encoder.layer.1.output.LayerNorm.weight')[R],3), np.round(base('bert.encoder.layer.1.output.LayerNorm.bias')[R],3))"
}
```


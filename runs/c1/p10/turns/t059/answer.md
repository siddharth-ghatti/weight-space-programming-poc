## reasoning summary

I'm noticing that layer 1's mean-only leave-one-out score is already 97.93, about as strong as layers 7 and 8. The jump at layer 9 stands out, so I'm wondering what changes there, possibly the carry channel. I'm excluding the residual dimensions in that calculation, but the carry heads at layers 2 and 6 may still affect things through the reserved channel.

I'm checking whether the reserved carry dimensions leak into the rest of the features through layer normalization: they sit in the same mean and variance computation as every other dimension, so their values could shift each token's normalized features. The shared shift per token is roughly the sum of the eight class scores divided by 768, so I'm wondering whether that could still carry some label information if the scores aren't centered.

I'm working through whether the residual-dimension cuts leave any path open, and the LayerNorm statistics seem like the only one left. Those statistics include the reserved dimensions, so their shared per-token shift and scale would be weak but nonzero. I'm checking whether layer 8 has anything special that would explain the jump at layer 9, and I don't see anything beyond those cuts.

I'm checking whether the layer normalization statistics could carry class information: the reserved scores enter the per-token mean and variance. If the ridge outputs sum to a constant across classes, the mean carries no label signal, but the sum of squared scores in the variance might, since confident predictions have larger norms.

I'm weighing whether the leak could be real: the reserved-dimension scores enter the per-token LayerNorm statistics, so their sum and squared norm might carry a faint signal, though I can't measure that directly. Comparing the L9 jump across 5-fold and LOO evaluations gives consistent results, so the next question is whether that jump reflects real signal or this leak, and a test on the edited model should settle it.

I'm wondering whether the L9 jump could be a leak rather than real signal: the carried scores in the reserved dimensions feed into the LayerNorm statistics, so in-sample ridge outputs might shift features at test time. If that's the case, test accuracy would fall below what the LOO numbers suggest. I'll check the magnitude of the reserved dimensions at L9 against the others and see how much they move the normalization.

I'm going to check how large the reserved dimensions are at layers 8 and 9, since if they're tiny, their effect on the layer-norm statistics would be negligible. I also want a cleaner test for leakage, possibly by seeing whether the leak signal tracks those carried scores, and I'll compare the layer 9 jump across the 5-fold and LOO evaluations.

I'm checking whether layer 8 has anything unusual that would explain the jump at layer 9, and the edits there look the same as in the neighboring layers, just cutting the reserved dimensions. That makes a layer-specific leak less likely, though the carry head from layer 6 is still something I want to account for.

I'm checking the magnitude of the reserved dimensions in the layer 8 and layer 9 hidden states, since that will show how much they could move the LayerNorm statistics. Layer 10 and later hold around 98.3 to 98.5, and layers 1 through 8 sit near 97.3 to 97.9, so the layer 8 output looks like a real shift in representation quality rather than an artifact.



Features hold around 97.3–97.9 LOO until layer 8, where they jump to 98.66 before settling near 98.3–98.5. Since the reserved dims carry leaked in-sample scores, I'm checking their magnitudes to see whether the L9 jump is a leak artifact.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nd=np.load('base_feats.npz'); y=d['y']\nfor L in [2,3,6,7,8,9,10]:\n    M=d[f'mean_hidden_L{L}']; C=d[f'cls_hidden_L{L}']\n    print(L, 'res |mean| %.4f  res sd %.4f   nonres |val| %.3f' % (np.abs(M[:,RES]).mean(), M[:,RES].std(0).mean(), np.abs(M[:,KEEP]).mean()),\n          ' sum_res mean/sd %.4f %.4f'%(M[:,RES].sum(1).mean(), M[:,RES].sum(1).std()), 'sumsq', (M[:,RES]**2).sum(1).mean().round(4))\n"
}
```


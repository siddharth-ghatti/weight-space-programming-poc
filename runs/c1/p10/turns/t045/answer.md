## reasoning summary

The relative differences look large: 12% after layer 0 and 36% after layer 1, which seems like a lot for only 512 of 3072 units pruned. Those figures are measured on mean_hidden, which includes the huge outlier dimensions, so the drift may be coming from them. On the subset, the edited model's LOO is 97.27 against 97.75 for the base, a drop of about 0.5%, and the full-data base at L9 is 98.7, so the edited full-data number should land near 98.

I'm questioning the 12% relative change after just the first layer with m=2560, since that seems too large for the units pruned, so the unit selection might have been poor. I'm weighing the design alternatives: the pooling layer costs a full layer's worth of parameters, and I'm considering whether the last kept layer could handle the CLS position directly so the pooler isn't needed.

I'm working out whether the last layer can double as the pooling step: since only the CLS output there matters, its attention heads can be rewired to average over all tokens, which would drop the separate pooling layer. The catch is that the layer-8 features look weaker than the layer-9 ones on the LOO check, so that trade-off needs testing.

Checking whether the last layer can handle CLS pooling directly: its other tokens don't matter, so its attention could be rewritten to average over all positions. But the per-token FFN nonlinearity can't be reproduced from CLS alone, so that shortcut loses the M9 features.

I'm thinking about the pooling layer: it costs a full layer's parameters, but its feed-forward units would sit unused. The real need is getting the mean-pooled features into the CLS position so the pooler can use them, and I'm checking whether those unused units could be put to some other use.

I'm weighing a leaner setup: fixing the depth at four layers plus the pooling layer would cut the total to about 52.8M, but I'm doubtful it can keep up with the deeper versions' accuracy.

I'm considering whether the pruning distortion could be reduced by fitting the pruned units' output with a least-squares reconstruction from the kept units. That would need intermediate activations I can't get without rebuilding the forward pass, which the rules rule out, so this route is probably closed.

I'm weighing how far to push the pruning: the subset result shows a 0.5-point drop at m=2560, which I'm trying to judge against the full-data baseline before deciding on the next size.

I'm checking whether the 12% drift after layer 0 is driven by the outlier dimensions, like dim 308, rather than by the pruning itself. I'm also considering whether the Gaussian proxy selection is picking poor units, and if so, whether a better selection criterion would hold the test accuracy above 97.8.

I'm considering whether the pruned units' contribution could be approximated better than with a constant bias: a linear term in x would be ideal, but the FFN output has no slot for it except through the kept units, so that route is blocked too.

I'm weighing a different size trade-off: keeping 10 layers at m=2560 gives about 87.4M parameters, while dropping one layer and keeping m=3072 gives about 88.2M.

I'm weighing whether a linear carry of M0 into the reserved dimensions could recover some accuracy, since M0 isn't affected by pruning and the earlier combination with M9 looked promising. The carry heads at layers 2, 6, and 10 already write ridge scores, so I'm checking whether that channel can hold the M0 term.

I'm wondering whether the layer-2 carry head could be rewritten to carry ridge scores fit on M0 instead of M2, since M0 is untouched by pruning. The open question is whether a linear M0 term adds anything beyond the M9 kernel, given that the reserved-dim carries from M2 and M6 already seemed to add nothing on top of it.

I'm checking how the Gaussian proxy was built, starting with the keys in ffn_proxy.npz, to see whether the unit selection itself is what's pushing the drift.

I'm checking how the Gaussian proxy was built, since the stats available are only mean and CLS hidden states, not token-level covariances. Maybe the token hidden states were used somewhere, but I haven't read them yet, so I'll look at the saved file's keys to see what it actually contains.



After pruning to m=2560, the after-layer-8 features score 97.27 LOO against 97.75 for the base, a 0.5-point loss with 12% feature drift already at layer 0. I'm inspecting the proxy file next to see how units were chosen and where the drift starts.



## tool_use: run_python

```json
{
 "code": "p=np.load('ffn_proxy.npz'); print({k:p[k].shape for k in p.keys()})\nfor k in p.keys():\n    a=p[k]; print(k, a.dtype, a.reshape(-1)[:5])\ns=structure(); print(s['layers'], s['m'])\n"
}
```


## reasoning summary

I'm comparing leave-one-out estimates across the model variants, with the M9 models scoring highest at roughly 98.5 to 98.7 percent. I'm treating LOO as a proxy for test accuracy, since the test set should be i.i.d., and noting that the original model's 98.31 percent test score suggests its LOO might sit around 98.1. The shallow models, with k=1 or 2, look like they land somewhat lower, near 97.5 percent.

I'm weighing the shallow models' estimated accuracy, around 97.8 percent, against the threshold, which would mean roughly even odds of failing and so isn't acceptable. The deeper M9 models sit at 98.5 to 98.7 percent, which is comfortable. I'm also considering whether further tuning of M0+M1 could push its best score up, though picking the top result from a grid search overstates it.

I'm considering a way to use the model's own activations to compensate for pruned FFN units, by routing the layer's intermediate outputs through a scaled selection matrix. That approach would only normalize per token and lose scale and offset, so it wouldn't compensate precisely, and it looks too hacky and expensive to be worth it.

I'm weighing whether dropping FFN units is worth it, since the proxy already lost ground at m=1024, with M9 falling from about 97.2 to 95.0 on the subset. The errors seem to build up with depth, so even m=2048 may not hold. Dropping whole layers might be a cheaper route, though I'd need data on which layers matter for the M9-level features.

I'm considering whether the pooling layer could be avoided by reading the [CLS] output instead, but the CLS features at layer 9 were weak, so I'm looking at whether layer 8's attention could be repurposed for the CLS position.

I'm looking at whether layer 8 can be repurposed for the CLS position: if its attention is set to uniform for every head, CLS would just pick up the mean of M8's hidden states, followed by layer 8's FFN. That would give M8-level features (about 98.0%) rather than M9 (about 98.6%). I'm also considering keeping layer 8's attention mostly intact and using one head to carry the class scores, though that would only allow a linear mapping from M8.

I'm comparing a few candidate configurations: the full layer stack with pooling at m=3072 keeps roughly 98.6% LOO at about 95.3M parameters, while dropping to layers 0 through 7 gives about 98.1% at 88.2M. Dropping one or two middle layers could land somewhere around 81 to 88M, though I don't know yet how much accuracy that would cost.

I'm weighing whether a two-layer model could reach the accuracy I need by giving the pooling layer's twelve heads several different poolings and letting its FFN act on the CLS position. The M0+M1 kernel sits at 97.9% with 38.6M parameters, so I'm checking whether richer features could close the gap to 98.2%.

I'm weighing whether a ~65% reduction is worth the risk, since a two-layer model would only reach roughly a 55 to 65 percent chance of passing. I'm concluding that I need about nine real layers to hold the accuracy target, so the remaining savings would have to come from pruning the feed-forward units or dropping whole layers.

I'm weighing whether forward-pass activations could be recomputed in numpy from token-level reads, but that would amount to reimplementing the layer's forward computation, which is off-limits. So I'm dropping that route and going back to data-driven importance measures for choosing which FFN units to prune.

I'm checking whether I can get the per-unit FFN statistics without reimplementing the layer, by building a probe model whose outputs expose them. Normalizing the output per token drops the scale I need, though, so this doesn't recover the unit-wise values cleanly. I'm now considering whether setting W2 to a diagonal-like selection could isolate individual units.

I'm considering whether I can get per-unit FFN importance without reimplementing the forward pass, by building a probe that writes the unit activations into the output dimensions. The normalization after the FFN would discard the absolute scale, though, so this only gives relative patterns, not clean importance values.

I'm thinking about recovering per-token FFN unit values by routing selected units into the residual dimensions and using two constant reference dimensions to back out the per-token normalization statistics. If that works, I could read the unit activations per token from the harness output without reimplementing the forward pass.

I'm dropping the idea of probing per-unit FFN activations, since reading them back would mean rebuilding the layer's forward pass or running many probes, and that's too costly. Instead I'm going to check whether the earlier pruning failure came from outlier dimensions, since the mean of the input activations is not zero, and whether the compensation step made things worse for layer 0's attention.

I'm checking whether any outlier dimensions have unusually large gamma or beta values, and I'm also weighing a second ranking that scores each unit by its output weight norm times its input weight norm scaled by gamma, leaving out the mean. The pruning failure at m=1024 looks fundamental, since removing two thirds of the units cuts roughly half the variance in each layer and those losses compound. Published work on post-training FFN pruning of BERT suggests accuracy collapses without some reconstruction step, usually layer-wise least squares on sample data, and I'm asking whether that could be done here without running the forward pass myself.

I'm weighing whether pruning FFN units or dropping layers can shrink the model without the accuracy loss of the two-layer option, since the feed-forward layers hold most of the parameters. Without reimplementing the forward pass, I don't have a reliable data-driven way to rank units, so I'm looking at whether a bolder layer-dropping option could land around 80 to 88M.

I'm sketching what a two-layer model could still hold: a carry head in the first layer for the M0 signal, the first layer's mean as M1, and the pooling layer's twelve heads computing several pooled views of that output. I'm also noting the pooling layer's feed-forward on the CLS position, with its 3072 kernel units and 768 tanh pooler units, and that the first layer could simply stay as the original layer 0.

I'm leaning toward the safer path of keeping layers 0 through 8 and pruning the feed-forward units more moderately, perhaps down to around 2048. I'd check that against the 97.17% baseline with a single 1024-example read before committing to it.

I'm comparing the single-layer pruning results with the compounded drop by layer nine, where the relative change in representations jumps to about 1.15. That size of shift suggests the compensation step may be pushing states off-distribution, so I want to check that before trusting the pruning numbers.

I'm checking whether the rel-change jump at layer 9 comes from a few outlier dimensions, so I'll compare the per-dimension mean shifts between the pruned and base runs at each layer.



Leave-one-out, the shallow layers reach 97.5–97.9%, right at the 97.8% threshold and too risky, the middle around 98.0–98.3%, and the deep layer about 98.6–98.7%. I'm going to check whether the failed proxy pruning was a systematic offset or a real loss of information before giving up on FFN width.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nF = np.load(WORK/'base_feats.npz')\nfor L in [1,2,3,5,9]:\n    Xb=F[f'mean_hidden_L{L}'][:1024]; Xp=np.load(WORK/f'pass_3/mean_hidden_L{L}.npy')\n    d=Xp-Xb; shift=d.mean(0); resid=d-shift\n    top=np.argsort(-np.abs(shift))[:6]\n    print(L,'norm base mean-vec',round(float(np.linalg.norm(Xb.mean(0))),2),'shift norm',round(float(np.linalg.norm(shift)),2),'resid per-sample norm',round(float(np.linalg.norm(resid,axis=1).mean()),2),\n          'base per-sample centered norm',round(float(np.linalg.norm(Xb-Xb.mean(0),axis=1).mean()),2))\n    print('   top shifted dims',top, np.round(shift[top],2), 'base vals',np.round(Xb.mean(0)[top],2))\n"
}
```


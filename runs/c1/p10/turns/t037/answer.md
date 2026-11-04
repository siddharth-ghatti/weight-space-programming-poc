## reasoning summary

Few dead units show up in layers 1–8, so pruning the FFN won't buy much for free. The likely final setup is k=10 with m=3072, which comes to about 95.3M parameters, roughly 13% smaller. I'm also wondering whether the pooling step can be simplified: the pooler only reads the CLS token, so maybe that token's state at the last real layer already works as a summary. I'm about to look at the CLS kernel at L9.

I'm weighing whether the last layer's attention could be flattened so the CLS output works as a pooled summary, but that would throw away layer 8's token-level processing, so it only gets back to the earlier 98% level rather than the full-depth result.

Weighing the two options, the bigger model (A) with about 95.3M parameters looks safer, since it keeps layer 8's token-level processing and should land near 98.66%. The smaller one (B) would drop to roughly 98.06%, with a real chance of failing the target. Unless a better option turns up, I'll go with (A).

I'm weighing a few more structural options: dropping layer 0, using a deeper layer's mean, or skipping only layer 7 before pooling. Dropping just one layer would likely land near the M8-level accuracy, so it wouldn't beat option (B).

I'm checking whether deeper layers, M10 through M12, beat M9 on the full set with the α-mix, since the earlier subset result hinted they might. If M9 holds up against them, I'll finalize option (A).

I'm weighing a more modest pruning step: trimming the FFN to m=2816 would drop about 256 units per layer and save another 3.9M parameters. The catch is that the proxy importance has been unreliable without compensation, so I'd need a 1024-subset check before committing to it.

I've got three spare passes for exploring. Mild FFN pruning would save only a few million parameters, with some risk. I'm weighing that against bolder alternatives, like trimming the FFN more selectively or reworking the attention-only path.

I'm considering whether the FFN input could be approximated from the residual stream without computing attention, since after the attention LayerNorm the output is largely driven by the residual. That would let me estimate per-unit importance from the hidden states I can already read, instead of relying on the Gaussian proxy.

I'm weighing an approximation for the FFN input that skips attention and just uses the layer-normed hidden state, since the residual stream dominates after attention. From there I could compute unit activations on real token states and rank units by that, which would be sturdier than the Gaussian proxy. Still, it leans toward simulating part of the layer's forward pass, so it sits in a gray area.

I'm weighing whether a data-driven importance estimate could justify pruning the FFN to around m=2048, which would save much more than the mild cut. But Exp 1 at m=1024 already did serious damage, so even m=2048 may hurt the features noticeably. The literature suggests post-training FFN pruning without reconstruction costs real accuracy, so the savings probably aren't worth the risk or the passes I have left.

I'm weighing a mild FFN prune to m=2560, which would save about 7.9M parameters, and checking it against the M9 subset baseline of 97.75%. I'd need to see roughly similar accuracy before adopting it. The 768-dim head on [0..7] plus pooling looks like it would land near 97.9%, so it isn't obviously better.

I'm checking whether compensating for the dropped units matters, since in the earlier experiment the shift in layer 1 was large. For layer 0, the proxy suggests most units are mostly inactive, so pruning there may be less harmful. Still, I'd want to estimate what the dropped units contribute on average, roughly summing their small negative GELU means across the 512 removed units.

I'm weighing a possible compensation term for the dropped units, estimating their small negative mean contribution. If the proxy holds, it helps a little, and if not, the harm is minor. Then I'd check whether this pruning option is worth one pass, since it would leave two passes for final reads and verification.

I'm planning to install the pooling layer in the pruned experiment X so its 1024-sample read captures the final CLS state, which would let me skip re-reading those samples if X is adopted. If X fails, I'd fall back to the unpruned model and need two passes for the final read, keeping the total within budget.

Picking m=2560 for the pruned experiment, since the larger cut at m=2048 risks losing too much accuracy based on what happened at m=1024. Expect the damage at 2560 to be small, maybe a few tenths of a percent.

I'm considering whether to put the pooling into layer 8 while keeping its attention only for non-CLS tokens, but that doesn't work as planned. What I actually need is the mean of layer 8's output over tokens, so I'm weighing whether the mean of its attention sublayer output could stand in for it.

I'm wondering whether layer 8's attention could be altered so the CLS row stays nearly uniform, which would let me approximate the pooled state without another full attention pass.

I'm asking whether only some of layer 8's heads could be made near-uniform, so the CLS attention broadens for every sentence without disturbing the other tokens.

I'm working out whether pooling could be folded into layer 8 itself, since if it is the last real layer, the other tokens' outputs stop mattering and I could reshape its heads freely. The catch is that CLS would then only see attention-weighted linear mixes of the layer-8 inputs, not the full contextual processing it currently gets.

Now I'm weighing whether the final CLS state could come from layer 8 instead of the full layer 9 pass, and I'm checking the leave-one-out accuracy of an α-mix on M8 against the all-layers head. I'll also look at M10 and M11 for comparison.



Few units in layers 1–8 are nearly dead, so mild FFN pruning is low-risk but gains little. Before settling the depth, I'm getting full-data leave-one-out results for the α-mixed features at L8, L9 and deeper.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef zsall(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef loo(Z, y, gammas=(0.2,0.3,0.4,0.6,0.8), lams=(0.0003,0.001,0.003,0.01)):\n    sq=(Z**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y]-1/8; out={}\n    for g in gammas:\n        K=np.exp(-g*D2); w,V=np.linalg.eigh(K)\n        for l in lams:\n            f=w/(w+l); Hd=((V**2)*f[None]).sum(1); Yh=V@(f[:,None]*(V.T@T))\n            out[(g,l)]=round(float((((Yh-Hd[:,None]*T)/(1-Hd[:,None])).argmax(1)==y).mean()),4)\n    return out\nfor L in [7,8,9,10,11]:\n    for a in [2,5]:\n        r=loo(zsall(LNr(a*F[f'mean_hidden_L{L}']+F[f'cls_hidden_L{L}'])[:,KEEP]),y)\n        v=np.array(list(r.values())); print(f'L{L} a{a}: best {v.max():.4f} median {np.median(v):.4f} min {v.min():.4f}')\n"
}
```


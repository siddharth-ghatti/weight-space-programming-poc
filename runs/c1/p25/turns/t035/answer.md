## reasoning summary

I'm weighing whether a hybrid approach would work, since the test pass bar sits about half a percent below the starting model. The deep path's CV of 98.1 looks close to the starting model's 97.8 linear, which is encouraging, and I'm now turning to how to cut the deep path down.

I'm now weighing whether pruning the FFN could cut the deep path's cost more cleanly than dropping layers. Keeping m=2048 might still hold up with only a small loss, which would bring the model to about 79.6M parameters at k=10. Pruning the FFN with a crude proxy may have been too pessimistic, so I'm considering a better one: least-squares reconstruction using inputs estimated from a single probe pass.

I'm checking whether least-squares reconstruction on the FFN could let me keep a wider intermediate layer, since m=1536 or 1024 would still cut parameters substantially. I'm also weighing whether the rule against reimplementing the forward pass covers computing the GELU activations from read inputs for the closed-form refit.

I'm now looking at compressing the FFN without reimplementing the forward pass: least-squares refitting on activations read from the probe model is a standard closed-form step. Still, I'd rather use the exact read activations than chain my own computations, to stay on the safe side of the rules. I'm also wondering whether clustering units by similar input weights and merging them could shrink the layer using weights alone.

Weighing the shallow option's odds: a 30M model would save the most parameters, but at k=6 the estimate lands around 97.4, which falls short of the bar. The deeper path's test score should land somewhere around 98.3 to 98.6, which clears it comfortably.

I'm ruling out the shallow option: its expected test score lands around 97.4 to 97.7, so it likely misses the bar. A deeper model at j=9 is the safer path, unless pruning the FFN can bring its cost down, which is what I'm turning to next.

I'm weighing whether exact probe activations would be worth the cost: nine layers would need nine probe passes, leaving little slack, and 64 sentences is too few samples for a least-squares refit of a wide FFN output matrix without heavy ridge regularization. So I'm checking how well a cheaper proxy from the existing token reads tracks the real layer inputs.

I'm checking whether the FFN's inputs can be estimated well enough from the existing token reads, since exact probe passes for nine layers would leave little slack. If the proxy tracks the real hidden states closely, a closed-form least-squares refit on the kept units stays within the rules.

I'm considering whether the attention output could be recovered from the layer's final hidden state. The sublayer structure makes that hard to invert cleanly, since the normalization step discards scale and shift. A single probe pass might give the exact attention-sublayer output for one layer, but I'm checking whether that's worth the cost.

I'm considering whether to shrink the FFN more gently, dropping the width to around 2048 with importance-based folding, since the 768 test came out poorly. That would cut about 16M parameters at k=10, but the CV noise on 1024 ids is roughly ±0.4%, so small differences will be hard to tell apart.

I'm trying to pick out which FFN units matter by their contribution to the pooled residual stream, since without gradients that's the only handle I have on class-relevant information. The sentence-level variance of each unit's mean activation looks like the signal to track, while token-level variance may matter more for what the later layers do with it.

I'm now trying a wider cut, m=1536, using the same importance scores, to see how far the test score drops. If the loss stays under about 0.3%, the model lands near 71.7M parameters; if it's closer to 1.5%, I'd fall back to m=2048.

I'm considering a stronger selection rule for the FFN units, scoring each by how much its output varies across sentences and how large its output weights are, rather than assuming a Gaussian. The input for that scoring would come from a proxy hidden state, with the attention contribution folded in through the normalization.

I'm going to try m=1536 with the same importance-based selection, since that tests how much width the FFN can lose. If the loss stays small, it cuts the model to around 71.7M parameters; if not, I'll fall back to m=2048.



The deep features cover varied semantic cases, so pooling won't target them; I'll go with the deep path and a moderate FFN cut across the 10 layers. Next I'm bisecting `m` with all 12 layers at m=1536 to see how much layer-9 features lose.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom prunelib import apply_ffn_prune\nprint(apply_ffn_prune(1536, list(range(12)), W, base, save_edit, set_structure, WORK))\nwith open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- Exp C result: layer dropping ~ quality tracks # layers kept (edited 7-layer skip stack ~ base L7). No free lunch.\n- Deep fixes are diverse semantic cases -> shallow pooling tricks unlikely to close gap.\n- Exp D (pass7): m=1536 all layers (imp + mean-fold bias edits on L0-10), read means 1024.\n\"\"\")\n"
}
```


## reasoning summary

I'm checking whether the deep KEEP dims in M9 through M11 are leaking, since carry heads in layers 2, 6, and 10 write into the reserved dims and the LayerNorm stats in later layers mix those in.

I'm weighing whether the M9–M11 KEEP dims are actually leaking through LayerNorm stats from the carry heads, but M7 would show the same leak if so and doesn't, so the jump at M9 looks like genuine depth. Since the original model's 98.31% supports that, I'm now looking at the trade-off: I need roughly 10–11 layers for accuracy, and the question is whether to drop middle layers or prune FFN width with compensation.

I'm weighing whether FFN compensation is even feasible, since I only have layer-output hidden states and can't isolate the attention output that feeds the FFN without running attention myself. That rules out closed-form refitting, so I'm looking at structural options instead: pick a layer configuration, read features, and fit a classifier on top.

I'm now weighing a structural experiment instead of FFN compensation: keep a subset of the early-to-middle layers, give each kept layer a reduced FFN, and end with the pooling layer. Features would come from the final CLS output plus mean-pooled hidden states, and then a kernel pooler gets fit on top. The cost is roughly 24.43M plus k times (2.37M plus 1537 times m), so k around 11 with a small m is what I'd be aiming for.

Looking at the parameter budget again: even ten content layers at full width plus a pooling layer comes to about 102M, so the full-width version doesn't help. Narrowing FFNs looks hard to do without losing the deep features, and simple magnitude-based selection likely does that too. Dropping whole layers might be the cleaner route, but each full layer costs about 7.09M, so I'd need to see which ones matter most.

I'm now considering whether I can get the FFN intermediate activations some other way, since the layer-output states don't separate the attention output that feeds the FFN.

I'm considering whether I can recover the FFN input by modifying layer l so its output exposes it, for instance zeroing the FFN and resetting LN2 to identity, though the normalization makes the h_attn values hard to invert cleanly.

I'm considering whether I can get the FFN input without running attention myself: set the LN2 scale and bias to identity so the layer output exposes the normalized attention sum, then temporarily reset LN1 so that output is that sum directly. From there the true FFN input follows from LN1's original gain and bias, and I'd compute the FFN activations from it.

I'm now asking whether computing the FFN unit activations from the recovered FFN input counts as reimplementing the forward pass, since a single linear layer plus GELU on read activations is a gray area. My earlier pruning scores came from per-unit statistics, so I'm checking how I actually produced them from the token hidden states.

I'm settling on dropping whole layers instead of narrowing FFNs, since I can't cleanly recover the FFN inputs without bordering on reimplementing the forward pass. The reduced model's features can then be read and the classifier refit on top, which keeps the size reduction substantial while staying within the rules.

I'm considering dropping whole layers instead of narrowing FFNs, since skipping layers may degrade the representation only gradually through the residual stream. Narrowing FFNs saves less per unit than it looks, because the per-layer FFN cost is large but dropping a layer removes the attention cost too. The real trade-off is best settled by experiment, and my remaining budget allows a small number of screening runs.

I'm planning a screening protocol: read mean hidden states at all layers for the first 1024 sentences in one pass, then compare kernel CV on those features against the base model on the same sentences. The final pooling layer will also need to carry the mean of the last content layer, and possibly M0 through the reserved channel, but I'll settle those details later.

I'm now deciding which layers to keep and at what FFN width, and I want a pruning criterion grounded in the activation statistics I already have. I'll check whether the earlier naive pruning at m=1024 really wrecked the deep features by comparing pass 5's per-layer mean features against the base model's on the same 1024 sentences.



Deep layers hold most of the accuracy, so I'll need about 9–10 content layers, and I'm now comparing pass 5 against base features on the same 1024 sentences to measure how much pruning costs.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels()[:1024]; B=load_base()\nP5=loadpass(5)\ndef bn(X): X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return max(max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,)).values()) for g in gs)\nfor l in [1,2,4,6,8,10,11]:\n    b=B[f'mean_hidden_L{l}'][:1024][:,KEEP].astype(np.float64)\n    e=P5[f'mean_hidden_L{l}'][:,KEEP].astype(np.float64)\n    # relative distortion\n    rel=np.linalg.norm(e-b)/np.linalg.norm(b-b.mean(0))\n    print(l, 'rel dev %.3f'%rel, 'base %.4f'%kcv(bn(b)), 'pruned %.4f'%kcv(bn(e)))\n"
}
```


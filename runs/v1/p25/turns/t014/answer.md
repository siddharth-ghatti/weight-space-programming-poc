## reasoning summary

I've found that unsupervised low-dimensional projections don't help, and stacking with leakage only helps under strong regularization. The exact joint concat ridge at 97.5% depends on carrying s0 = A0·M0 with its combination held fixed, so I'm planning to carry those 8 numbers forward and compute s11 = A11 from there.

I'm weighing whether the carried s0 can be kept exact and still fit the scale of s11 without perturbing M11. Using a strongly regularized stacking, with lam0=300 and a refit, is the other option at 97.2%. I'm also looking for a way to sidestep perturbation entirely, since the final CLS only sees LN-transformed pooled attention outputs that are linear in M11.

I'm now thinking about a leakage-free refit: if the s0 term is fixed at some α and I refit A11 on the same training points, the in-sample s0 values are nearly perfect, so the residual targets shrink and A11 gets pulled in the wrong direction.

I'm checking whether refitting A11 on the edited features with targets Y − A0·M0 stays consistent with the joint ridge, since A0 and A11 were fit together in-sample and that fixed point seems to rule out extra leakage. The open question is whether the s0 actually carried in the model matches that exact A0·M0 term.

I'm checking whether projecting out the u' directions before fitting A11' keeps the refit free of leakage, and how a separately scaled readout of those directions would recover s0' as roughly s0 times some scale factor κ.

I'm considering whether the carried s0 can be made exact with a fixed scale, or whether a few refit scalars on the edited features would be enough. Refitting A11 with the joint ridge fixed point looks consistent, but the carried s0 may differ from the true A0·M0. To check how clean the carry is, I want to examine the write and read spectra of the u-directions across layers 1 through 10.

I'm considering a CLS-only carrier instead: one token holds s0, so perturbation to the others only flows through attention to CLS, and perhaps no head needs sacrificing in layer 0. The open issue is how CLS in layer 0 can collect the mean of the token states at all.

I'm checking whether the carried s0 can be written exactly through a single head's value and output projection, but A0 would have to lie in the row space of that head's value matrix, which generally it doesn't.

I'm looking at whether the carried s0 can go into the W_o projection across all twelve heads, since their stacked value matrix is likely invertible. That would let CLS write A0·M0 exactly, but other tokens would pick up a small component along the u-directions too, which I'd need to check.

Since every token could carry the u-components through a modified layer-0 output projection, the pooled vector at layer 11 would hold something close to a reweighted mean of h0 rather than exact M0. That's an approximation, but a workable one if the attention weights average out roughly uniformly. I'd want to check how far that reweighting drifts from M0 before relying on it.

I'm thinking about using identity attention in layer k so each token reads back its own hidden state h_j through a head's value projection, which would let the carried s0 be computed from the exact per-token states rather than from an unreadable post-attention state.

Looking at whether a head can be sacrificed in layer 0 to make the carried s0 exact, versus using a CLS-only broadcast with its query zeroed so the uniform mean is applied in every head.

I'm looking at making the CLS query zero in every head, so its layer-0 attention is a uniform mean over all tokens, while other tokens only pick up a small perturbation through the shared query change. The write-up then needs the output projection modified to add the u-directions, which would also nudge the other tokens along those directions, and I'd have to check how big that effect is.

I'm weighing a CLS-carrier idea too: since CLS's attention output is an invertible linear map of M0, the full M0 information could enter CLS at layer 1, though I'd expect it to degrade through the later layers. I still think carrying the u-directions is the more controllable route, so I'm turning to analyze which residual directions layers 1 through 10 leave alone.

I'm looking at how each layer reads the residual through its LayerNorm, so a direction u gets read as gamma times u over sigma before any weight matrix acts on it. I'm also checking how the attention and FFN outputs write back into the residual, since that determines how the u-component carries across a layer.

I'm working out how each layer's LayerNorm rescales a component along a direction u, which means the direction gets multiplied elementwise by the gamma weights and then divided by the norms. So the inert directions have to be tracked layer by layer rather than fixed once. I'm also starting to look at how much each layer reads from the residual through its attention inputs.

I'm checking how each layer's LayerNorm and read/write matrices treat a candidate carrier direction, since the u-component gets rescaled by gamma products at every layer. The goal is a subspace that layers 1 through 10 barely read or write, so the carried value stays intact until layer 11.

I'm working out a joint objective over layers 1 through 10: each layer's LayerNorm rescales the candidate direction through the gamma products, so I'd propagate u forward layer by layer and measure how much each layer reads from it and writes into it. Choosing the starting direction to minimize that total energy would turn into a generalized eigenproblem, with the write terms tied to the readout vector at each layer.

I'm weighing whether the energy-minimizing subspace is worth the effort, since the gain looks like about 1% at best. I still need to account for the base activity along those directions, because the final readout would pick up that background as noise. The u-component written at layer 0 has to stand out from it while staying small against the total token norm.

Trying out a richer readout now: letting CLS in layer 11 pool with attention patterns beyond the uniform mean, such as concentrating on the opening tokens of the sentence. I'll run a baseline Design 1 check first, then see whether the carry-forward approach is worth the extra work.

I'm checking whether a shorter carry distance works just as well, since layer pairs like [6, 11] and [0, 11] both sit around 97.3–97.5. Fewer layers between the write and the readout should mean less corruption along the carried directions.

I'm weighing whether a shorter carry distance helps, or whether pooling at layer 6 and feeding forward is cheaper. The layer pairs are close enough that I'd want the simplest route to a correct readout. I'm also considering a completely different setup: concatenating a mid-layer pooled vector with the layer-11 CLS, pooling the mid-layer piece into CLS directly so nothing has to be carried.

I'm weighing whether the sacrificed-head carry is worth its cost, and whether a bag-of-tokens readout from a mid layer could be pooled straight into CLS instead. Next I'm checking whether CLS in layer 11 could pool from specific token positions, such as sentence-initial tokens, rather than the uniform mean.

I'm considering whether CLS's own attention and FFN updates in the later layers would corrupt a carried value, and whether overwriting CLS with a large pooled signal at layer k could keep it intact through to layer 11.

I'm checking whether CLS's own attention and FFN updates would corrupt a carried value, and whether a strong constant marker in CLS could keep the info intact. The linear FFN term would just distort the info linearly, so it might survive as long as the combined map stays invertible.

I'm considering whether CLS could act as the carrier: its attention and FFN updates would act roughly linearly on the stored info if CLS is dominated by a constant marker, so the info would evolve as a linear map layer by layer. The open question is whether others' attention to CLS would shift, which would depend on setting the marker near the base CLS state.

I'm weighing whether CLS could carry the info itself by pinning it to a constant marker close to its base mean state, so its own attention and FFN updates act roughly linearly on the stored value. The harder part is CLS's query, which would otherwise mix sentence-dependent attention into it, so I'm considering forcing uniform attention, though that brings in deep pooled features too.

I'm weighing whether to lean on the CLS token itself as the carrier, or to use one sacrificed head at an early-to-middle layer to broadcast 8 class scores forward, with the final layer doing the pooling. Next I'd evaluate the expected noise from the read/write analysis, then run two passes over the 1496 examples: one reading mean_hidden at layer 11 to measure how much the perturbation shifts it, and one checking whether the carried scores come back out intact.

I'm considering whether to sacrifice a head at layer k for the carry, or to reduce it more gently by zeroing only its query so it attends uniformly. That changes its output for every token, so its normal function is lost either way. The better option might be to pick a head whose base attention is already broad and close to uniform, so the change stays small, and to test both the M11 effect and how cleanly the u-carry survives in one edited read.

I'm weighing whether to sacrifice a head at layer k for the carry, and I'm leaning toward replacing the eight least important value dimensions of that head with the carried scores. That would partly break the head, so I'm considering instead choosing whichever head already attends most broadly, since its normal behavior would change least. Checking attention entropy across heads on about 64 texts would tell me which one to pick.

I'm weighing which layer to tap for the carry. Layers 0, 1, 3, and 6 all land near 97.3–97.5, and tapping at layer 6 would mean only five layers of processing between the write and the readout, so that looks like the most attractive option for keeping the carried signal intact.

Now I'm working out a read/write subspace analysis across layers 1 through 10, looking at which directions each layer's attention and FFN actually read from and write to, to find the most inert ones for carrying the scores.

I'm computing the singular value spectra of each layer's attention read matrix to see how small the smallest values are compared to the largest and median, since a ratio around 0.01 would leave room for carrying the scores.



Stacking the supervised class scores gets to about 97.2%, so a few of them carried from a shallow layer might add 1–1.5 points. Before building that, I'm checking the singular spectra of the attention, FFN and write matrices for residual-stream directions that layers 1–10 barely use.



## tool_use: run_python

```json
{
 "code": "def g(n): return W(n)\nfor l in [1,4,7,10]:\n    p=f'bert.encoder.layer.{l}.'\n    R=np.concatenate([g(p+'attention.self.query.weight'),g(p+'attention.self.key.weight'),g(p+'attention.self.value.weight')],0)\n    s=np.linalg.svd(R,compute_uv=False)\n    W1=g(p+'intermediate.dense.weight')*g(p+'attention.output.LayerNorm.weight')[None,:]\n    s1=np.linalg.svd(W1,compute_uv=False)\n    Wo=g(p+'attention.output.dense.weight'); so=np.linalg.svd(Wo,compute_uv=False)\n    W2=g(p+'output.dense.weight'); s2=np.linalg.svd(W2,compute_uv=False)\n    q=lambda s: np.round([s[0],np.median(s),s[-50],s[-10],s[-1]],3)\n    print(l,'QKV',q(s),'W1g',q(s1),'Wo',q(so),'W2',q(s2))\nfor l in [0,3,6,9,11]:\n    p=f'bert.encoder.layer.{l}.'\n    for n in ['attention.output.LayerNorm.weight','output.LayerNorm.weight']:\n        gm=g(p+n); print(l,n,np.round(np.percentile(gm,[0,5,50,95,100]),3), 'argmin',np.argsort(gm)[:5])"
}
```


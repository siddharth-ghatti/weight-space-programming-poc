## reasoning summary

I'm noticing the mean absolute residual grows from about 0.04 at layer 3 to 0.19 at layer 6. My guess is that each layer norm subtracts a shared mean, pushing the same offset onto all eight reserved dimensions, so the offsets pile up. I'm now checking whether the growth holds once the values are centered across those dimensions, comparing the layer 6 to layer 3 ratio.

I'm checking the centered values again: the L6 vector is exactly proportional to the L3 one, with about 4% drift across three layers, so maybe 6–10% over the full stack. That's a mild multiplicative noise on the linear term, and I'll treat it as small. I'm also planning to zero the writes from the existing carry heads into the reserved dimensions.

I'm now laying out the full model: layer 0 gets one head repurposed as a carry head with uniform attention, its value dimensions projecting the carried features into the reserved slots, while the other heads write nothing into those dimensions.

I'm checking whether the carried signal survives the reserved-dimension pollution from the residual stream at layer 0. Since the layer-0 attention norm mixes in the embedding's reserved coordinates, the carried value becomes the fitted linear term plus a fixed contribution from those coordinates, which the joint fit can absorb.

I'm working out how the carry heads can deliver the signal without the layer-0 residual pollution spoiling it, and I'm planning to pre-compensate the carry projection so the reserved-coordinate contribution from the embedding cancels out. I'm also considering that the averaging over tokens should keep this approximately correct, so a simple adjustment should be enough.

Checking whether the reserved-dimension pollution from the layer-0 residual stream matters: the embedding coordinates are around 0.5 to 1, so I'm weighing them against a carry magnitude of about 1 on a 768-dimensional vector. I'll pre-compensate the carry projection and fit against the measured values rather than trying to eliminate the pollution at the source, since the embeddings are frozen.

I'm settling on a concrete plan: first fit the carry weights w from the joint kernel-plus-linear regression on the pruned layer-9 features and the layer-0 features, then work out the rest from there.

I'm planning to refit the carry weights on all 1496 ids once the new features are read, and the reserved-dimension values can be predicted from the old carry outputs and the rescaling factor, so the readout can be refit without rerunning the whole pipeline.

I'm settling on the carry weights from the joint regression instead of an M0-only projection, since the residual-aware fit should beat ridge scores on M0 alone, and I can refit those weights later without rerunning the pipeline.

I'm settling the final score as a kernel term plus a carried linear term, with the weights and the kernel coefficients fit jointly. I'm now checking whether the reserved dimensions in the pool layer can carry that linear term directly, so the scales would line up without extra linear units.

The reserved-dimension output after the pool layer comes out to roughly the carry term scaled by (1+c) and the layer-norm factor, plus the kernel scores added in the FFN. So the final score is a mix of the carried linear piece and the kernel piece, both landing in the same reserved slots. I'm checking how the attention norm scale σ1(x) affects that balance.

I'm working out how to fit the joint model, since the carry weights w are needed before the readout can be fit on the FFN-input features, and each depends on the other.

I'm weighing whether to fit the carry weights against the residual-scaled features, so the linear term matches what the FFN actually receives after the per-sample rescaling. Then I'll set up a provisional carry projection and the pooling attention with centering as the first step.

I'm estimating the per-sample scale from the measured reserved-dimension values, since the carried vector should be proportional to the fitted projection, with some small error from the per-token pollution averaging out.

I'm working out how the carry head's output combines with the pooled embedding's reserved coordinates, which both scale with the same per-sample factor. My plan is to estimate that embedding contribution from a read with the carry turned off, then set the carry weights to the fitted values minus that contribution.

I'm trying to avoid pollution from the layer-0 residual stream, and the plan is to make the carry signal large enough to dominate it. Before settling on that, I want to check how big the reserved-dimension values in the layer-0 hidden state actually are.

I'm weighing whether the per-sample scaling of the reserved dimensions can be absorbed as a varying bias, since the constant part of the layer-0 embedding doesn't vanish once it's rescaled. I'm considering whether a head in layer 0 could zero out the embedding's reserved coordinates so only the carried signal remains.

I'm going to check whether the layer-0 reserved-dimension pollution can be cancelled by a self-attending head, but that looks unreliable. Instead I'll measure how large the pollution is, comparing its variation across samples to the carried signal, and then decide whether to absorb it or make the carry gain big enough to dominate.

I'm checking whether a carry value around 3 per dimension would be large enough to matter. Across the 8 reserved dimensions that adds about 72 to the sum of squares against roughly 768 for the whole vector, so the layer-norm scale would shift by only about 5% and do so the same way for every token in a sentence.

I'm weighing whether to shrink the reserved-dimension pollution in layer 0 by reading layer 1 instead, since zeroing those coordinates at the layer-0 output norm might leave only the carried signal.

I'm considering whether to move the carry to layer 1 instead: zeroing the reserved dimensions in the layer-0 output norm would leave h1 clean there, so the carry head in layer 1 writes only the signal. That would mean cutting the reserved columns in layer 1's FFN so the carried values don't leak forward.

I'm now looking at the layer-1 option as a way to avoid the layer-0 reserved-dimension pollution, since zeroing those coordinates at layer 0's output norm would leave the layer-1 features clean for the carry head. The open question is whether the linear term from the pruned layer-1 features holds up as well as the layer-0 version, so I want to compare them on the pass-19 data before committing to that change.

I'm weighing whether to sacrifice a layer-1 head for the carry instead, since zeroing the reserved dimensions at layer 0's output would leave the carry head there free of pollution. Picking which head to drop is the open part: I can't measure its contribution without reading attention patterns, and weight norms might not reflect its role.

Comparing the layer-2 carry head's linear score against the layer-0 version: 97.46 for the pruned layer-2 features versus 97.81 for layer 0, which suggests the carry could read from the already-sacrificed head's features without losing much.

Checking whether the layer-1 features with the pruned network give a linear score close to the layer-0 version, and whether re-pruning with drift correction would also help the layer-2 features.



The carried signal survives the LayerNorm chain, but layer 0's carry head mixes with the embedding's reserved dims, so I'll zero those dims and compare the layer-1 linear term against layer 0 on the pruned features.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz'); yy=y[:1024]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nE9=cn(np.load('pass_19/mean_hidden_L9.npy')); Ke=np.exp(E9@E9.T-1)\nfor nm,X in [('M0',d['mean_hidden_L0'][:1024]),('M1e',np.load('pass_19/mean_hidden_L1.npy')),('M2e',np.load('pass_19/mean_hidden_L2.npy')),('M1base',d['mean_hidden_L1'][:1024])]:\n    C=cen(X); print(nm,{c:krr_cv(Ke+c*C@C.T,yy,[0.005],k=10,seeds=range(5))[0.005] for c in [0.25,0.5,1.0]})\nE9b=cn(d['mean_hidden_L9'][:1024]); Kb=np.exp(E9b@E9b.T-1); C0=cen(d['mean_hidden_L0'][:1024])\nprint('base K9+lin0 on 1024:',krr_cv(Kb+0.5*C0@C0.T,yy,[0.005],k=10,seeds=range(5)))\nprint('M0 res-dims std across samples', d['mean_hidden_L0'][:,RES].std(0).round(3), 'KEEP median std', np.median(d['mean_hidden_L0'][:,KEEP].std(0)).round(3))"
}
```


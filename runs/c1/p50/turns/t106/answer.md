## reasoning summary

I'm weighing the joint linear M0 model: the base is 98.46, and M2 comes in at 98.35. On the pruned 1024 subset, adding the linear M0 term lifts the score from 96.91 to 97.81, and for 1536 it goes from 97.15 to 98.09. Against the 1024 baseline, K0*K9 at 97.95 sits close to these.

The 1536 carry lands at 98.09, above the 1024 product-kernel baseline, so carrying from M0 looks like the strongest option. The open question is how to get that signal into the model: a carry head at layer 0 would cost one of its content heads, and layers 0 and 1 currently use the reserved dimensions for ordinary content.

I'm considering whether M0 can be read from anywhere other than layer 0's embeddings, since higher layers don't carry h0 directly. One idea is to repurpose a layer 0 head that contributes little, though I'd need a way to judge that. Separately, layers 0 and 1 would have to stop reading and writing the reserved dimensions, or their content would corrupt the carry, at the cost of the 8 embedding dimensions there.

I'm checking whether the reserved dimensions in layer 0's embeddings have higher variance than the others, since that decides how much cutting them out of layers 0 and 1 would cost. If they're not outliers, dropping those 8 dimensions should only shift the downstream features a little.

I'm checking whether a carry head in layer 0 could write into the reserved dimensions without disturbing content, since the residual stream keeps h0's reserved values and I can't cut them with the weights alone. One option I'm considering is a head that attends to each token's own value, so it can subtract that token's embedding from the reserved dims.

I'm still on the layer 0 attention sublayer: zeroing the reserved dimensions' gamma and beta there would wipe out the residual and attention contributions, but the FFN is per-token and couldn't compute the carry, so I'm weighing whether layer 1's carry head could take over from there, which would put the layer 1 reserved-dim features back in play at about the M1 base of 98.35.

I'm checking whether layer 0's attention LN could wipe the reserved dims, with one head repurposed to carry r, though that means losing that head's content contribution. The other layer 0 heads would stop writing into the reserved dims, and the cost compared to M1's base of about 98.35 is still open.

I'm noting that the residual h0 in the reserved dimensions would pollute the carry, but after mean pooling that pollution is just a linear function of M0, which the readout already absorbs. The per-token LayerNorm scaling makes that only approximately linear, though.

I'm working out a simpler route: if layers 0 through 8 never touch the reserved dimensions, the pooled values there would only reflect the layer-0 embedding scaled by the LayerNorm factors. That wouldn't give the jointly fitted projection, though, so a carry head with value weights set to the projection w is still needed. I'm starting the layer 0 design by making one head the carry head, with its query weights zeroed.

I'm working out the layer 0 carry head's weights: uniform attention over non-padding tokens, with its value projection mapped into the reserved dimensions while every other head and the bias leave those dimensions untouched.

I'm working out the layer 0 and layer 1 changes: cutting their reads and writes of the reserved dimensions, including the FFN input and output weights and the LayerNorm settings, so that only the carry remains there. I still need to check how the residual carries h0's embedding content in those dimensions past the attention sublayer.

I'm working out how the reserved-dimension values pool after layer 0: the carry, the mean-pooled embedding coordinates, and a per-token LayerNorm offset. The common offset is the same across all 8 reserved dims, so mean-subtracting across them should cancel it, leaving the carry plus the embedding term.

I'm working out how the carry gets read out: the reserved coordinates from the embedding add a fixed linear term to the pooled values. I could fit that term by measuring it with a known carry, but the per-token normalization factors make it only approximately linear in M0.

I'm wondering whether the reserved dimensions in the embedding output are close to constant, since that would keep the leftover embedding values from polluting the carry. If they are, the pooled values there would mostly reflect the carry itself.

I'm checking whether the embedding LayerNorm gamma is small at the reserved dimensions, which would make the residual contribution nearly constant and harmless. I'm also noting that keeping the same reserved dimensions means the cuts in layers 2 through 8 stay as they are, so the base features are preserved.

I'm working out the pollution from the residual: the embedding's reserved coordinates, after the LayerNorm scaling, add a fixed linear term in M0 that just sums into the carry. If that term is known, I can offset the carry head's projection by it so the combined value still comes out as w. The scale factors act the same on both pieces once they share the layer-0 attention output, so the compensation should carry through.

I'm settling on the carry head's value weights: since the token-specific residual adds a per-token term, I'll compensate by subtracting the reserved-coordinate selector from the projection, so the pooled total still comes out as w. The per-token scale factors only make this approximate, so the final ridge fit has to absorb the remainder.

I'm looking at how the reserved-dimension measurements feed into the final classifier. The plan is to measure the carried 8-dim vector per sample after installing the layer-0 carry, then fit a matrix D on top of it jointly with the basis weights, since the carried values only approximately follow the intended linear term. I still need to work out how that D term gets fit.

I'm planning to have the pool layer's reserved dimensions carry the measured c(x), then use attention LayerNorm and an FFN to remix those coordinates linearly through D, with a pair of GELU units per output.

I'm weighing a layer 0 carry head against the existing layer 2 one. The layer 2 carry already reaches the base at about 98.35, and on the pruned 1024 subset it trails the layer 0 linear baseline (97.46 versus 97.81), so the layer 0 option looks worth the cost of changing layers 0 and 1. I'll need to re-evaluate the features once that change is made.

I'm considering whether any existing layer-0 or layer-1 head could double as the carry without losing its function, but a uniform-attention carry head needs its own dedicated slice of values, so there's no way to share it. I'm also looking at layer 2's current carry head and layer 6's, wondering whether the reserved dimensions could survive layers 0 and 1 untouched instead.

I'm weighing which layer 0 head to repurpose as the carry, since a uniform-attention carry needs its own slice of values and can't share one with an existing head.

I'm checking whether any layer 0 head already attends near-uniformly, since that would be the cheapest one to repurpose as the carry, though I'd need to measure its attention pattern to know.

I'm weighing whether a weight-only heuristic, the layer 0 head with the smallest W_O·W_V norm, is a cheap enough way to pick which head to repurpose as the carry, rather than measuring attention patterns with another pass. I'm also wondering whether the layer 0 FFN has to drop its reads and writes on the reserved dimensions, which would mean the layer 1 changes are needed too.

I'm now checking which layer 1 writes and reads touch the reserved dimensions, and whether the layer 1 output LayerNorm needs to pass them through unchanged. I'm also asking whether the existing carry heads in layers 2 and 6 should be switched off so they stop adding their own scores into the reserved coordinates.

I'm weighing whether to zero the existing carry heads' writes into the reserved dimensions in layers 2 and 6, so only the new layer-0 carry populates them. Layer 0's FFN input would also pick up the carry and the other reserved-dimension writes, and I still need to work out how cutting those reads affects it.

I'm now weighing what the layer 0 FFN and layer 1 lose by cutting their reads of the reserved dimensions, since the layer 1 output norm wipes those coordinates anyway, so only the layer 0 FFN and layer 1 reads actually matter.

I'm weighing how to set up the layer-0 carry: the plan is to fit the reserved-dimension weights jointly with the base M0 and pruned M9 features, then check whether the layer-0 and layer-1 cuts disturb the pruned features enough to need a refit. I'll decide on the feature width first, since that determines the rest of the budget.

Next I'm planning the pool layer: attention pooling, FFN kernel units, and linear units for the carry, then the pooler decode and classifier. After that I'll read the pooled and FFN-input activations for the 1496 ids over two passes, fit the readout, and verify with three predict passes. I'm also weighing whether 1024 or 1536 kernel units fits the remaining budget.

I'm going to test whether a finite set of GELU anchor units can approximate an exact RBF kernel fit, using the normalized pooled features. I'll check it offline on the base features, with a random 1000-point training anchor set and a sweep over the scale and bias parameters, plus the linear M0 term.

I'm checking whether a finite set of GELU units with anchors can match the exact kernel fit, so I'll test the scale and bias settings offline on the base pooled features before committing to the layout.



Kernel (pruned M9) plus the jointly-fit linear M0 reaches 97.8% at m=1024 and 98.1% at m=1536, beating the base product kernel's 97.95%, so the plan is a layer-0 carry head writing M0 scores into reserved dims. Next I'll verify offline that a finite GELU-anchor feature map matches exact kernel ridge regression before building it.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nZ9=cn(d['mean_hidden_L9']); C0=cen(d['mean_hidden_L0'])\ndef anchor_cv(a, b, nanch=1000, lam=1e-2, c0=0.5, seeds=(0,1), k=10, lamlin=None):\n    accs=[]\n    for s in seeds:\n        f=folds(y,k,s); S=np.zeros((N,8)); rng=np.random.RandomState(s)\n        for i in range(k):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            anc=rng.choice(tr, min(nanch,len(tr)), replace=False)\n            Phi=gelu(a*(Z9@Z9[anc].T)+b)\n            F=np.hstack([Phi, np.sqrt(c0)*C0*np.sqrt(Phi.shape[1]/C0.shape[1])*0+np.sqrt(c0)*C0]) if c0>0 else Phi\n            mu=F[tr].mean(0); Fc=F-mu\n            Wt=np.linalg.solve(Fc[tr].T@Fc[tr]+lam*np.eye(F.shape[1]), Fc[tr].T@(Y[tr]-Y[tr].mean(0)))\n            S[te]=Fc[te]@Wt\n        accs.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(accs)),4)\nprint('exact KRR K9+0.5lin0:', krr_cv(np.exp(Z9@Z9.T-1)+0.5*C0@C0.T,y,[0.005],k=10,seeds=(0,1)))\nfor a,b in [(1,0),(3,-1),(5,-2),(8,-4),(4,0)]:\n    print(a,b,[anchor_cv(a,b,lam=l) for l in [0.1,1.0,10.0]])"
}
```


## reasoning summary

I'm noticing that the kernel on M9 plus a linear term on M0, fit together, gets the same 98.37 as the joint model, so the jointly fitted linear part seems to be what helps. Stacking out-of-fold linear scores didn't, and I suspect that's because the joint fit uses the full 767-dimensional linear function, regularized alongside the kernel.

I'm working out that the jointly fit linear term on M0 can be reduced to 8 scores, one per class, which a carry head could compute and store in the reserved dimensions. The idea is that these weights come from the joint fit with the kernel, so they capture what the kernel misses, unlike stacking. I'm now checking how cosine-normalizing Z0 affects that weight matrix.

I'm working out how the carried scores would be computed on the unnormalized mean of the layer inputs, since the cosine normalization doesn't carry over. The open question is how the final layer norm would scale the reserved dimensions, and whether an early carry at layer 0 would survive the intermediate layers without being disturbed.

I'm checking whether the carry could come from layer 0 instead, but that would mean changing how reserved dimensions behave in layers 0 and 1, which also alters the content there. Layer 2 is simpler since its carry head already exists, so I'll test K9 + lin2 and K9 + lin1 against the joint fit. Then I'll look at how the pooling layer handles the reserved dimensions when it computes the kernel features.

I'm settling the scale question: the FFN normalizes its input per sample, so the kernel features will line up with the cosine-normalized vectors if the dataset mean is removed first. The kernel readout then fills only the reserved rows of the output projection, leaving the other rows at zero, so the reserved dimensions end up holding the carried linear term plus the kernel scores.

I'm checking how the pre-LN residual carries the reserved dimensions through: the attention output at CLS may add a scaled copy of the carried values, which the first LayerNorm then normalizes by the sigma of the pooled vector. After that, the FFN's kernel scores get written into the same reserved dimensions before the final LayerNorm.

I'm weighing how the linear part should be computed inside the FFN: the per-sample normalization of r by σ1 makes the carried scores scale differently from the kernel scores. A way around it is to realize the linear function exactly with GELU units, since GELU(z) − GELU(−z) = z, which would take two units per output, so eight outputs need sixteen units.

I'm working through how the carried linear term would be combined with the kernel readout in the final layers. The per-sample normalization in the layers means the carried values from unnormalized M2 get scaled by a sample-dependent factor. So I'll test the scaling by using centered, unnormalized M2 with a global constant, and check the effect on the final reserved dimensions.

I'm checking how the residual carries these values through, since the attention output may add a scaled copy and the first LayerNorm divides everything by the pooled vector's sigma. That means the linear term ends up scaled by 1/σ1, which varies per sample, so it's roughly the normalized M2 rather than an exact match, and I could fit against that scaling directly.

I'm noting that attention in the pool layer carries the reserved dimensions too, so the CLS value scales r by a factor of (1+c) before the first normalization divides it by σ1. That makes the linear part a scaled copy of the carried values, and I'm considering fitting features of the form (M2 − mu)/σ1 jointly with the kernel features, with separate penalties.

I'm checking whether the carry at layer 2 survives the pruned layers, since the earlier edit cut reads and writes of the reserved dimensions in layers 2 through 10. The open question is whether the FFN refit in the pruning step kept those cut rows at zero or quietly reintroduced writes into them.

I'm checking how the reserved dimensions get scaled by each layer's normalization, since the carried value at h9 depends on a per-token, per-layer factor from the layer norms and gammas. I'm also confirming that the cut reads and writes in the pruned layers still hold, so the carried values stay intact through to the pool layer.

I'm checking how the per-token layer-norm factors scale the carried reserved dimensions, since that makes the effective feature the carried linear term times a per-sample factor. I'll measure that factor on the h9 tokens directly rather than assume it.

I'm measuring how much the carried value gets scaled per sample, since each token's copy is multiplied by its own factor across layers and the mean over tokens gives a per-sample scalar. The per-dimension constants are fixed, so they can go into the weight matrix, and the open part is that per-sample scalar.

I'm checking whether the carried values would still be scaled per sample by the layer norms, so I'll measure that per-sample factor on the h9 reserved dimensions before fitting anything. Installing a carry head first would change those readings, so I'll read them from the current carry instead.

I'm weighing whether to carry the linear term from layer 2 at all, since the per-layer normalization makes its scale depend on each sample. Carrying from layer 8 instead would mean fewer normalization layers in between, and the pool layer itself could be the place to read it, though it only sees the mean of the final content layer.

I'm checking whether the layer-2 carry is worth keeping at all: if a linear term from M2 or M1 adds to the kernel on M9, I could repurpose the existing carry head there. I'll test that on the base features first, then on the pruned network's M0 and M2 features.



An additive kernel plus linear model fit jointly reaches 98.37%, nearly matching the joint product kernel, so a jointly fit linear term could fit in the 8 reserved dims via the carry head. I'm checking which early layer's linear term works best, on both base and pruned features.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz')\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X, Xref):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nZ9=cn(d['mean_hidden_L9']); K9=np.exp(Z9@Z9.T-1)\nfor l in [0,1,2,3,6]:\n    Zl=cn(d[f'mean_hidden_L{l}']); Cl=cen(d[f'mean_hidden_L{l}'],None)\n    r={}\n    for c in [0.25,0.5,1.0]:\n        r[c]=krr_cv(K9+c*(Zl@Zl.T),y,[0.005],k=10,seeds=(0,1,2))[0.005]\n    r2=krr_cv(K9+0.5*(Cl@Cl.T),y,[0.005],k=10,seeds=(0,1,2))[0.005]\n    print('M%d cosnorm-lin'%l, r, ' unnormalized-lin c.5:',r2)\n# pruned features (1024) \nyy=y[:1024]\nfor tag,pk in [('m1024',19),('m1536',22)]:\n    E9=cn(np.load(f'pass_{pk}/mean_hidden_L9.npy')); Ke=np.exp(E9@E9.T-1)\n    Z0=cn(d['mean_hidden_L0'][:1024])\n    out={'K9e':krr_cv(Ke,yy,[0.005],k=10,seeds=range(5))[0.005], 'K9e+lin0':krr_cv(Ke+0.5*Z0@Z0.T,yy,[0.005],k=10,seeds=range(5))[0.005]}\n    if pk==19:\n        Z2=cn(np.load('pass_19/mean_hidden_L2.npy')); out['K9e+lin2e']=krr_cv(Ke+0.5*Z2@Z2.T,yy,[0.005],k=10,seeds=range(5))[0.005]\n    print(tag,out)"
}
```


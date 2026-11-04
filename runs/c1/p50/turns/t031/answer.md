## reasoning summary

Checking the Nyström setup: with 768 random anchors I get 96.66 against 97.53 for full KRR, so roughly a 0.9% loss. Using all 1196 training points as anchors would probably close that gap, but the pooler only has 768 units to work with.

I'm weighing whether the exponential kernel's advantage over linear comes mainly from its quadratic terms, which would make an explicit polynomial expansion attractive, though 768-dimensional quadratic features are too many. I also suspect the kernel ridge needs essentially all training points, since at such a small regularization it is close to interpolating.

I'm weighing a simpler route: keep the original features (the carried 8-dim linear scores and the layer-11 pooled CLS) but cut cost elsewhere. Pruning layer 11's FFN doesn't help much since that loses accuracy, so a deeper model stays expensive. A middle ground of about five or six full-FFN layers, roughly 60M parameters, looks like the realistic budget, and a shallow model of three or fewer layers hurts accuracy.

I'm weighing whether to test attention-pooling designs empirically, since that's the lever I haven't explored yet. Improving the head with FFN units in the pooling layer is another option, though each design needs careful construction and passes to evaluate.

I'm sketching the pooling layer's feed-forward units as anchors: each one a GELU of a scaled dot product between the normalized CLS vector and a training point, with the output projection writing KRR readout weights into the eight score dimensions.

I'm sketching how the pooling layer's FFN could hold a full KRR readout: each unit a GELU of the normalized CLS vector against a training anchor, with the output projection writing scores into the eight dimensions while the attention layer norm zeroes those dimensions out first.

I'm checking whether the anchors can be the training points' own normalized activations, so the FFN units compute the kernel row for each input. That would put the full readout inside the pooling layer without a separate step.

I'm checking whether the GELU kernel on the normalized activations behaves like a usable kernel. In its left tail GELU decays with a Gaussian shape rather than exponentially, so the kernel's behavior far from the anchors is different from the exponential one I'd been assuming.

I'm testing whether a GELU-shaped kernel, GELU(a(s − s0)) on the cosine similarity of the M0 features, can stand in for the RBF kernel in kernel ridge regression, and I'd hope for around 97.5% if it works. I'm also considering pairs of units per anchor with different thresholds, which would approximate a piecewise-linear kernel.

Now I'm working out the readout inside the pooling layer: the FFN units would compute kernel values against the training anchors, and the solved coefficients would go into the output projection. Two GELU units per anchor with different thresholds should approximate an exponential-like kernel, since sums of ReLUs can approximate any convex increasing function. The solution is a 1496×8 coefficient matrix, and the FFN sums those weighted kernel values into the eight score dimensions. Cross-validation of this setup should match exact KRR, which is the part I want to confirm.

Weighing the parameter budget now: the pooling layer's FFN could take about 1500 units, one per training anchor, which brings the model to roughly 29M parameters. That is close to the two-layer option with m=1, but it moves the kernel readout into the pooling layer where it gets 1500 anchors instead of 768 random ones. The question is whether it can beat the 97.8% that M0M1 RBF reaches under cross-validation.

I'm thinking about whether the attention layer in the k=1 setup can capture more than the mean of the embedding outputs. The CLS term is constant so it adds nothing, but several attention pools with different queries over h0 might pick up order, such as a pool that favors early positions, or pools keyed to specific words.

Checking whether a sparse anchor set could work instead of all training points: with the 768-unit budget, the Nyström loss looked too large, so fewer anchors would likely hurt more. The k=1 layout at about 1500 units looks cheaper overall than two pooling-related layers with full anchors.

I'm checking whether using every training point as an anchor, rather than a random subset, keeps the readout accurate. Next I'm working out how to get the FFN input u to be the normalized mean of the embedding outputs, with uniform attention and an output projection that undoes the mean.

I'm working out how to fix the output layer's bias and the first-layer weights so the centered mean feeds the normalization, which should behave like cosine normalization for the kernel. The final residual then combines the input with the FFN output, and I'm about to decide how to reserve the eight score dimensions in the layer norm scale.

I'm setting the eight score dimensions' LN scale and shift to zero before the layer norm, so the normalization stats come only from the rest of the vector. Then I'm checking that the FFN scores stay moderate, which keeps the output layer norm near unit variance and the readout close to the solved coefficients.

I'm checking how the pooled output's 8 score dimensions come out after the final layer norm: the shared mean and scale shift all eight equally, so the argmax is unchanged. That means the pooler and classifier can just pass those dimensions through as identity, with the pooler scaled small so it stays in tanh's linear range. I'm also noting that adding the linear ridge scores into the FFN output is an option.

I'm settling on a single GELU unit per anchor, so the kernel becomes GELU(a·s + b) on the centered, cosine-normalized M0 features, and I want to see how its cross-validated accuracy compares with the exponential kernel.

I'm considering solving the readout α by ridge on the feature matrix F = κ(S), which is what the model computes anyway, or equivalently by closed-form KRR, α = (K + λI)⁻¹Y with K = F, since the anchors are the training points. Next I'll test GELU-shaped kernels using the cosine similarity of the centered M0 features.



The 768 random anchors are the bottleneck, so I'm moving the kernel machine into the last layer's FFN, where each of the 1,496 training sentences serves as an anchor, and I'll check which GELU shape works best next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\ngelu=lambda t: 0.5*t*(1+erf(t/np.sqrt(2)))\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef cosnorm(X):\n    Z=X-X.mean(0); Z=Z-Z.mean(1,keepdims=True); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\nZ=cosnorm(M(0)); S=Z@Z.T\nprint('s pct', np.percentile(S[np.triu_indices(len(S),1)],[1,10,50,90,99]))\nL=(1e-3,3e-3,0.01,0.03,0.1,0.3)\nprint('exp', krr_cv(np.exp(S),y,L,k=10,seeds=(0,)))\nprint('exp2', krr_cv(np.exp(2*S),y,L,k=10,seeds=(0,)))\nfor a,b in [(4,0),(4,-1),(6,-1),(8,-2),(3,-0.5),(10,-3),(5,0)]:\n    print('gelu',a,b, krr_cv(gelu(a*S+b),y,L,k=10,seeds=(0,)))"
}
```


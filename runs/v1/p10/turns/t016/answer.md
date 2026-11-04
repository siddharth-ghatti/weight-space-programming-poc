## reasoning summary

I'm picking eight dimensions to use as the test vector, and I'm noting that dimension 308 is a large outlier, with a magnitude of about 185. That outlier may dominate the layer-norm standard deviation, so I'm trying to estimate its per-layer contribution, which lands near 8, though the sign is still unclear.

I'm working out whether the layer-norm standard deviation is below 1, since a lone outlier dimension near 8 might leave the overall spread small and cause growth. I'm treating the calibration as empirical, then turning to the layer-0 FFN neurons to pick the eight with the smallest output norm. For the projection, I'm leaning toward fitting a joint ridge on the m0 features rather than m1.

I'm deciding to zero out the layer-0 dimensions D in the LN1 gain and bias so they contribute nothing before the FFN write, and to zero those rows of the projection too. The projection reads from x, which is roughly the normalized embedding plus attention output, so I'll scale it so the per-token spread of the projected values lands near 0.5, estimating that from the sentence-level m0 features and expecting token-level spread to be larger.

I'm setting the bias around 10 so the GELU stays in its linear range across the projected values, which should keep the output exact up to float32 precision.

I'm settling the layer-0 FFN surgery: the eight stolen neurons get an offset bias so their output reduces to the projected values, their original write weights to the other dimensions get zeroed, and the protected dimensions stay untouched through layers 1 through 10 before the final readout.

I'm setting the final layer's attention so the queries are zeroed, which makes every token's attention weight uniform, and I'm making the value and output projections an identity scaled by a large constant so the mean over tokens dominates the CLS state.

I'm setting the final layer's output so the CLS state reduces to the layer-norm of the mean, with the feed-forward weights zeroed and the output norm set to identity. Next I'm looking at how to fit the pooler and classifier from the layer-12 CLS hidden state.

Setting up the classifier now: the pooler rows carry a scaled ridge projection through tanh, and since tanh is monotonic, the argmax of the logits survives as long as the values stay out of the saturation range.

I'm now zeroing the query, key, and value weight columns for the protected dimensions in layers 1 through 10, so they stay untouched through the stack.

Now I'm zeroing the protected dimensions in the layer-0 LayerNorm and FFN output, so they stay clean through the stack.

I'm setting the layer-0 intermediate weights so the eight chosen neurons read the projected directions with zeroed protected columns and a fixed bias. Then the output projection maps those neurons into the protected dimensions with a negated bias, so each one returns the projected value. Next I'm finishing the layer-0 output LayerNorm gain for the protected dimensions.

I'm checking how the magnitude of the protected dimensions changes across layers, and planning to set each layer's gain from that observed ratio.

I'm weighing whether to set the gain on the protected dimensions to the typical gain, since without any added signal those dimensions would otherwise shrink relative to the rest at each layer norm.

I'm weighing the size of the scale factor for the protected dimensions: too large and they would distort the layer-norm statistics for everything else, too small and the signal fades, so I'm leaning toward a modest gain below the typical spread and checking the magnitudes empirically as I go.

I'm estimating the token-level spread of the layer's hidden values, which I put near 0.7 once the outlier dimension is included, and expecting the layer-norm input to land around 0.8 to 1.0. I'm starting to settle on a gain for the protected dimensions based on that estimate.

I'm picking a small initial scale for the protected dimensions, around 0.05, with the gain near 0.8, since growth or shrinkage by a factor of about 100 would still be manageable. Even so, a tiny signal at the final layer may lose precision when the features are standardized in float32.

I'm going to measure the per-layer magnitudes of the protected dimensions in this first pass, then recalibrate the gains for each layer norm before building the final model. That means roughly two more passes over the 1024 examples, plus a pass over the remaining 472, which still fits the budget. I'm also weighing whether fitting the projection on all 1024 labels leaks into the evaluation, though I haven't settled that yet.

I'm going to evaluate the final model on the 472 held-out examples, since they were never used in fitting, and use that as the honest comparison against a mean-only baseline by dropping the protected dimensions.



I'm picking the 8 least important dimensions, D, and repurposing the 8 weakest layer-0 FFN neurons to write into them. Next I'll build the edit, zeroing D's reads and writes in layers 1–10 and setting up layer 11 for uniform attention.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]\nm11 = load('mean',11); m0 = load('mean',0)\nJ = np.concatenate([m11,m0],1)\nWf,b = ridge_fit(J,y,0.4)\nA = Wf[768:].copy()          # 768x8 applied to h0-like vectors\nD = np.array([222, 601, 495, 88, 254, 444, 233, 42])\nA[D,:] = 0\nz = m0@A; print('sentence-level z std per dim', z.std(0).round(4), 'mean', z.mean(0).round(3))\nnp.save('A_proj.npy', A); np.save('D_dims.npy', D)\n"
}
```


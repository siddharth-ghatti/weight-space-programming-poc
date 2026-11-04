## reasoning summary

The RBF kernel came out at 98.2 against 98.1 for the GELU version, a steady gap of about 0.1 that I'm content to accept. Next I'm considering whether the pooler could reproduce the RBF exactly, or whether a hybrid could approximate the exponential, but neither looks workable. Using more units per center, with GELU applied on both sides, is the remaining option I'm weighing.

I'm working out whether the kernel's linear and quadratic parts can be balanced by mixing GELU(u) with a scaled GELU(-u). Subtracting a portion of GELU(-u) lowers the quadratic-to-linear ratio toward the exponential's, and two units per center still fits within the 3072-unit budget.

I'm settling on GELU for the kernel and setting λ to 0.01, since the larger value looked a bit more stable across the folds. The 20-fold and 5-fold numbers point in slightly different directions, but the differences sit within noise, so I'm not going to tune further.

I'm now checking float32 precision: the alpha values reach about 5.6 in magnitude, so the partial sums of the GELU terms could run into the hundreds, with rounding error near 1e-3. I'll also look at how the training and cross-validation margins are distributed, to see whether any samples fall close enough to the boundary to matter.

Still writing the weight matrices...

Next I'm sorting out the final layer-norm and pooler weights, and whether the pooled output scale of 2 is right for the centered targets.

Setting the pooler's output scale at a=1 rather than 2, since tanh(2) is already near saturation for the centered score differences. The rest of the layer-norm and pooler setup carries over unchanged, and the remaining pooler rows aren't used by the classifier.

Next I'm checking the classifier weights to confirm they only read pooled outputs 0 through 7, and then I'll compare the expected logits for all 1496 ids against a read pass.



GELU-kernel KRR scores within about 0.1 of RBF, with modest coefficients, so I'm going with κ(u)=GELU(u) and λ=0.01. Next I'll check the z-score sd range, the classifier and pooler rows, and the layer-11 LN2 parameters before writing weights.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nX1=np.load('X1_full.npy'); F=X1[:,KEEP]\nsd=F.std(0); print('sd quantiles', np.quantile(sd,[0,0.01,0.5,1]).round(4), 'mean abs', np.abs(F.mean(0)).max().round(3))\nprint('outlier dims by |mean|', KEEP[np.argsort(-np.abs(F.mean(0)))[:5]], np.sort(np.abs(F.mean(0)))[::-1][:5].round(2))\ncw=W('classifier.weight'); cb=W('classifier.bias'); nz=np.nonzero(cw); print('classifier nz', list(zip(*nz))[:10], cw[nz][:8], cb)\npw=W('bert.pooler.dense.weight'); pb=W('bert.pooler.dense.bias'); print('pooler rows0-7 norm', np.linalg.norm(pw[:8],axis=1).round(3), pb[:8])\nprint('LN2 g res', W('bert.encoder.layer.11.output.LayerNorm.weight')[RES], 'LN1 g res', W('bert.encoder.layer.11.attention.output.LayerNorm.weight')[RES])\n"
}
```


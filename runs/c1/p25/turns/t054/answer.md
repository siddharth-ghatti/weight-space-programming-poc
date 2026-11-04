## reasoning summary

I'm noting that random tanh features don't beat the linear baseline, which is 97.37, and the random-feature version does worse. The quadratic kernel reached 97.82, and I'm checking how its diagonal entries come out with λ=1e-2, since the cosine-based Q(x,x)=1 seems to give a diagonal of 2. Now I'm turning to the kernel cross-validation.

Next I'm working out how the quadratic kernel's Q² term behaves. Its feature space is huge, so the random tanh features with 760 units can't capture it, and I'm weighing whether a Nyström-style setup with the training points as centers could stand in for it.

I'm working out how to get a g(t) = t + t² term per training point, which needs about 1496 units, so the 3072-unit FFN would fit it. I'd pair two GELUs per center to form an even quadratic-like response, and the output dense layer would write the weighted sum of those responses into the 8 reserved class dimensions. The dual coefficients would come from a numpy kernel ridge fit on activations read from the CLS hidden state.

I'm checking whether computing the kernel from read activations counts as legitimate kernel ridge, since the FFN only approximates g(t) with GELU, so the fitted coefficients may not match the implemented function exactly. In-sample accuracy should be near perfect with small λ, so the real question is whether the held-out estimate still holds once that approximation is in place.

I'm considering whether a kernel that GELU computes natively, K(x, x_i) = GELU(a·t_i + b), could replace the exact quadratic kernel, so the fitted coefficients match what the FFN actually computes. The catch is that this function might not be positive semidefinite, though kernel ridge on the resulting features could still work.

I'm checking whether computing the feature matrix from read activations counts as legitimate closed-form ridge, since the forbidden thing would be simulating the model instead of running real forward passes. I think designed nonlinear features of activations are fine for the fit, and I'll verify the result with actual forward passes on the practice ids.

I'm now sketching a pool-slot architecture for the kernel ridge model: the CLS token gets a uniform attention mix of the sequence, and the layer's FFN would hold the per-center nonlinearities.

I'm stuck on the cosine normalization: the FFN can compute a dot product with each center vector, but dividing by the norm of the z-scored vector isn't a linear operation, so the cosine term Q can't be computed directly from u.

I'm settling on a kernel defined directly on the un-normalized activations, using per-dimension scaling with a fixed constant instead of a per-sample norm, so the feature is linear in u and can be implemented in the FFN. The norms will vary across samples, so I'll need to check cross-validation for this variant, which my notes put at around 98.0 for the z-scored kernel without L2 normalization. I'm also planning how the FFN output layer maps each unit to the 8 reserved class dimensions with its dual coefficient, plus an output bias.

I'm checking whether LN2 stays close to identity, since the reserved dims carry both the linear scores and the scaled kernel output and should remain small relative to the other dims.

I'm checking whether the common scaling of the reserved dimensions after LN2 leaves the argmax unchanged, since the decode subtracts the mean over the 8 dims. The real concern is the relative scale between the carried linear scores, which vary by about 10% per sample, and the FFN kernel scores, because both are summed into the same dims before normalization.

I'm settling on the M2 and M6 carried terms: they can enter the fit as measured linear features scaled by their κ factors, alongside the kernel expansion. I'm also looking at whether the pooler's tanh can be applied to the reserved dims after centering over the 8 classes, so that it stays consistent with the decode.

I'm working out how the kernel cross-validation will match the M8 result. The plan is to read the normalized activations for all 1496 points, with the pool-slot LayerNorm scaling set so the reserved dimensions stay moderate and the tanh pooler doesn't saturate into ties.

Next I'm reading the cls_hidden states from the final layer and the mean_hidden states across all layers in the same pass, so every point's u comes from one forward run.

Working through the weight layout for the FFN units now: each one gets a first-layer row built from the centered, scaled center vector, with the bias absorbing the mean offset. I'm restricting those rows to the non-reserved dims and zeroing unused units out of the 3072 so they contribute nothing.

Now I'm weighing k=9 against k=10, and I want to test the GELU kernel on the z-scored M8 features before reading real activations. I'll sweep the scale a and offset b to see how the cross-validation holds up.

I'm weighing whether a GELU-based kernel, GELU(a·t + b), could stand in for the exact quadratic one, and I'm noting that for small arguments with b = 0 it expands to roughly a linear term plus a quadratic term, so it behaves much like the kernel I already want.

Testing the GELU kernel's small-argument expansion: it behaves like a linear term plus a quadratic term, close to the exact quadratic kernel I want, though with larger arguments it deviates. I'll sweep the scale a and offset b and compare cross-validation against the quadratic baseline.

I'm now testing the GELU-based kernel on the z-scored M8 features without L2 normalization, sweeping the scale and checking how the cross-validation holds up against the quadratic baseline.



Random tanh features fall short of the quad kernel (97.37, below the linear readout), so I'm switching to an exact kernel machine in the pooling slot's FFN, using GELU to approximate the linear-plus-quadratic kernel, and I'll check CV on z-scored M8/M9 next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, y, lams, seeds=(0,1,2)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=K[np.ix_(te,tr)]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\nLk={j:z(M[j])@z(M[j]).T/760 for j in (2,6)}\nfor top in (8,9):\n    Z=z(M[top]); T=Z@Z.T/760\n    print(top,'t diag mean %.2f sd %.2f offdiag range %.2f..%.2f'%(np.diag(T).mean(),np.diag(T).std(),T.min(),T.max()))\n    for a in (0.5,1.0,1.5,2.5):\n        for b in (0.0,):\n            K=gelu(a*T+b)/(a/2)\n            print(' a',a,'b',b,kcv(K+Lk[2]+Lk[6],y,(3e-3,1e-2,3e-2,1e-1)))\n    Q=T+0.5*T**2; print(' quadnoL2', kcv(Q+Lk[2]+Lk[6],y,(3e-3,1e-2,3e-2,1e-1)))"
}
```


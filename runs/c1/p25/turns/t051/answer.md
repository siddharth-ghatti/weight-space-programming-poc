## reasoning summary

I'm noticing the residual at j≥7 points to s6 having a different scale from s2, which makes sense since it's added later and passes through fewer layer norms, so I'll allow a separate κ per component. The scores are very small in spread, so a modest gain should keep them from disturbing the normalization. The κ variation runs roughly 7–12% in relative terms, and since the kernel CV assumed exact linear terms, I expect the modulation to have only a small effect on accuracy.

I'm planning to check the κ modulation in the CV simulation. For the final readout, the carried reserved dimensions at the final CLS should be κ2·s2 + κ6·s6, so I could re-fit the carry head weights to my joint-fit weights and measure κ2 and κ6 once. That measurement is a two-component regression per sentence, with eight equations and two unknowns each. Then I'd build the joint-fit features from those κ-scaled terms plus the kernel features, though the pool slot would still change κ at the end.

I'm now planning to read the final CLS and pooled outputs for all 1496 examples, then measure κ2 and κ6 at the last layer from the reserved dimensions. After that I'll run the joint fit and write the new carry-head, pooler, and classifier weights, keeping the reserved magnitudes near their starting scale so the layer norms stay unaffected. The pooled features still need care, since the reserved-dimension decoding depends on the new carry weights.

I'm working out how the reserved-dimension readout should decode into the pooler: a near-linear tanh mapping with a small gain, so that the classifier can combine those κ-scaled carry scores with the kernel features.

I'm checking how the z-scored carry features map back onto the raw M2 terms, since the centering shifts everything into the per-class bias. That bias can then be absorbed into the carry head's b_v, so the ridge fit only has to handle the scaled κ-modulated features.

I'm checking whether the carry dimensions match the mean over all non-padding tokens, including CLS and SEP, since the attention is uniform over them; the earlier regression already matched at those indices. I'm also wondering whether the reserved dimensions of the first-layer output are zeroed by the layer norm, and whether the value weights on those reserved dimensions need to stay at zero.

I'm checking whether the first-layer reserved dimensions are zeroed by the layer norm, and whether the value weights on those dimensions at layer 6 need to stay at zero so the carried signal doesn't mix in. I'll keep them at zero, then test a ridge fit on tanh-mapped non-reserved features plus the linear M2 and M6 terms, using an approximate normalized pooled state in place of the exact one.

I'm weighing whether the pooling layer should stay as it is or be replaced with a uniform-attention slot where the output is roughly α times the mean of the tokens plus the CLS residual, with the FFN turned off. A larger α would let the mean dominate the CLS's own state, so I'm checking the layer-11 weights to see how the current pooling behaves before deciding.

I'm going to make the uniform-attention pooling slot's mean term dominate by using a larger scaling factor, since the CLS contribution looked slightly worse in the earlier checks. The attention averages over every token, including CLS and SEP, so the reserved dimensions carry through the same way, and I'm setting each head's value block to a scaled identity with the output projection kept as identity.

Now I'm checking how the uniform-attention pooling normalizes its input, since the layer norm will divide by roughly α times the spread of the mean vector, which is smaller than a single token's spread. I expect the reserved-dimension values to be measured after that step rather than assumed.

I'm sketching the pooler features as tanh units over the standardized final hidden state, with a gain and bias per unit, and checking whether 760 of them are enough in a numpy design before deciding whether the FFN in the pooling slot is worth using for extra capacity.

I'm weighing how to set up the pooler units: random Gaussian directions, PCA directions, or training examples as centers. A tanh of a linear projection, tanh(a·t + b), has a quadratic term when b is nonzero, so I can reach both linear and quadratic pieces of the kernel from those units.

I'm going to test three feature schemes numerically on an approximated pooled state: random Gaussian directions with random biases, random training points as centers, and PCA-based directions.

I'm going to score the design on held-out data by adding the linear M2 and M6 terms, to see how close the approximation gets to the 97.82 kernel result.



The carried scores are tiny and each reaches the final CLS through a per-sentence factor that varies 7–12%. I'm checking in numpy whether about 760 tanh features can match the quad kernel before building the pooling slot.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef pcv(F, y, lams, seeds=(0,1,2)):\n    # primal ridge, features already scaled\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                mu=F[tr].mean(0); Ft=F[tr]-mu\n                A=np.linalg.solve(Ft@Ft.T+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=(F[te]-mu)@(Ft.T@A)\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\ntop=8\nZ=z(M[top]); Zn=Z/np.linalg.norm(Z,axis=1,keepdims=True)*np.sqrt(Z.shape[1])  # norm sqrt(d)\nlin26=np.hstack([z(M[2]),z(M[6])])/np.sqrt(760)\nrng=np.random.RandomState(0)\nfor nunits in (760,):\n    for a in (0.5,1.0,2.0):\n        Wr=rng.randn(nunits,760)/np.sqrt(760)\n        b=rng.randn(nunits)*1.0\n        P=np.tanh(a*(Zn@Wr.T)+b)\n        for sc in (0.03,0.1):\n            F=np.hstack([P*sc, lin26])\n            print('rand a',a,'sc',sc, pcv(F,y,(1e-2,3e-2,1e-1)))\n# pure linear reference: Zn linear + lin26\nprint('lin only', pcv(np.hstack([Zn/np.sqrt(760), lin26]),y,(1e-2,3e-2,1e-1)))"
}
```


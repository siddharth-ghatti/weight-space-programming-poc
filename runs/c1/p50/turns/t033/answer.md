## reasoning summary

Pair-even with a=1 plus a linear term reaches 97.79% at 10 folds on M0 alone, beating the exponential kernel (97.53%) and linear (96.3%). The kernel decomposes as S + q(S), where q is roughly quadratic, so polynomial-2-style kernels look like a good fit for M0. A degree-2 polynomial kernel on bag-of-embeddings lands around 97.6-97.8% at 10 folds as well.

I'm checking whether the 20-fold results hold for this kernel and for M0M1 as well, then working out how the linear term could be built from GELU pair units in the FFN.

I'm working out how the linear term could come through the attention-LN instead of separate FFN units: the reserved dimensions of the pre-LN vector could carry linear scores via the output projection. The per-sentence normalization scales those scores, but since the kernel already works on normalized features, the linear term stays consistent with it.

I'm settling on the simplest way to get the linear term: 16 GELU pair units in the FFN, since the 8 reserved dimensions of the normalized vector are zero and can't carry it directly. I'm also checking whether routing the scores through the pre-LN vector would shift the normalization enough to matter, and it should be negligible if those scores are small against the rest, which I can verify by reading the actual vector from the model.

I'm reconsidering whether the pair structure is needed: one GELU unit per anchor gives GELU(a·s) = a·s/2 + q(a·s)/2, so the quadratic part still comes through while the linear part folds into the linear term. That leaves the model as a sum of single-unit anchor terms plus 16 pair units for the linear weights, which fits within 3072.

I'm checking whether a single GELU unit per anchor can reproduce the quadratic part of the kernel, with the linear part absorbed into the free linear weights.

Single GELU unit per anchor, plus 16 pair units for the linear weights, should express the same solution. That brings the count to 1512 units, and I'm checking the parameter total from there.

Checking the parameter count: 1512 units gives about 29.1M parameters, roughly 26.6% of the original. The kernel's sensitivity to the scale a is the open question now, since a=1 looks best but a smaller a only pushes the kernel toward something like S + S², which scored a bit lower.

Next I'm going to compare kernel performance on M0, M1, and the concatenated M0M1 features, using 10-fold and 20-fold runs with a few seeds, and check whether the attention-pooling features add anything on top.



Pairing GELU terms reaches 97.79% at 10-fold on bag-of-embeddings alone, and the full solution fits in 1,512 units, about 29.1M parameters for a 1-layer model. Next I'll check accuracy more carefully across 10- and 20-fold runs, several seeds, and richer feature sets.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\ngelu=lambda t: 0.5*t*(1+erf(t/np.sqrt(2)))\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]; Cc=lambda l: d[f'cls_hidden_L{l}'][:,KEEP]\ndef cosnorm(X):\n    Z=X-X.mean(0); Z=Z-Z.mean(1,keepdims=True); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef kern(Z,a=1.0): S=Z@Z.T; return S+(gelu(a*S)+gelu(-a*S))/(a*a)\nL=(0.01,0.03,0.1)\nfor name,bl in [('M0',[M(0)]),('M1',[M(1)]),('M0M1',[M(0),M(1)]),('M0M1M2',[M(0),M(1),M(2)]),('orig',[M(2),M(6),M(10),Cc(12)]),('M0..M12',[M(l) for l in [0,3,6,9,12]])]:\n    Z=cosnorm(blockn(bl)); K=kern(Z)\n    print(name,'10f',krr_cv(K,y,L,k=10,seeds=(0,1,2)),'20f',krr_cv(K,y,L,k=20,seeds=(0,1)))"
}
```


## reasoning summary

I'm ruling out scaling up the kernel, since larger inner products would need a different beta or lambda, and at 0.5 nothing changed, so I'm dropping that. I think the feature set is near its ceiling. The remaining options are adding more data, where the learning curve still rises and the sentence weight of 0.3 helped a little, or trying token-level pooling variants, which look risky and costly. Next I want to check the learning curve more precisely.

I'm weighing whether more data is worth it: I'd run a cross-validation on 50%, 75%, and all of the practice set to see if the learning curve is still steep, though my sentences fit the distribution less well. I'm also noticing some practice errors look like label-convention problems, which is fine since test items near practice items inherit those labels. The most valuable move may be making sure the submission is robust before trying a token-level max-pool probe, which would cost about three passes.

I'm sketching the probe's head layout: eight uniform-attention heads carrying the top-512 principal components of the layer-11 mean, with the remaining four heads' outputs mapped back into the residual space. The catch is that the residual stream is already filled by the CLS token and the mean term, so folding the max-pooled features in alongside them would mix signals in the same 768 dimensions.

I'm trying to keep the 768-dimensional residual stream from mixing signals: the uniform-attention heads would carry the mean through the top-512 principal components, and the max-pooled features would need to go into the orthogonal 256-dimensional complement. The obstacle is that the CLS term fills every dimension, so I'd need a head that attends only to CLS and subtracts its component in that complement.

I'm weighing whether to drop the CLS term entirely by scaling up the attention output so the residual contribution becomes negligible. That would lose the CLS signal except for its 1/T share in the uniform average. Before deciding, I want to check whether a kernel on the mean-pooled layer-11 features alone holds up against the combined version, since linear results were 0.969 versus 0.975.

Dropping the CLS term would free the 768 dimensions for attention output, with 512 for uniform principal components and 256 for other pooling. I'd also want to check whether mean-pooled features alone hold up, since CLS is already a pooled representation. For the remaining four heads, I'm considering whether an exclusion-based mean or class-conditioned soft-max pooling would add anything beyond the uniform mean.

I'm considering whether max-pooling over tokens could capture key phrases regardless of sentence length, using the top directions of the hidden states as the basis. I'd need to estimate the token-level spread first, which would take another read pass over a small sample, and I'm not sure the gain would justify the cost.

I'm weighing whether the errors are really about pooling or just ambiguous wording, and I think the gain from a token-level probe is speculative and likely small. Writing more targeted data looks similarly modest. Next I want to measure the learning curve on the practice set alone, training on 50%, 75%, and 100% of the folds, to see how much more data would actually help.



Stretching the discriminant subspace doesn't help, since it stays flat at γ=0.5 and breaks the kernel scale above that. Next I'm checking how steep the learning curve is, to decide whether more hand-written training sentences are worth it.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=len(y)\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\ndef stdz(X): return (X-X.mean(0))/(X.std(0)+1e-6)\nZ0=stdz(ln(d['cls11']+2*d['mean11'])); K=gelu(Z0@Z0.T/768)\nfor frac in [0.4,0.6,0.8,1.0]:\n    accs=[]\n    for seed in range(6):\n        rng=np.random.default_rng(seed); f=rng.permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            trall=np.where(f!=j)[0]; tr=rng.choice(trall,int(frac*len(trall)),replace=False); te=np.where(f==j)[0]\n            T=np.eye(8)[y[tr]]; b=T.mean(0)\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+0.01*np.eye(len(tr)),T-b); P[te]=K[np.ix_(te,tr)]@A+b\n        accs.append((P.argmax(1)==y).mean())\n    print(frac, int(frac*1197), round(np.mean(accs),4), round(np.std(accs),4))\n"
}
```


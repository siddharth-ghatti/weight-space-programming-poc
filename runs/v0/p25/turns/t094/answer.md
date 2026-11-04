## reasoning summary

I'm checking whether the "w/ fit" variant is leaking: its in-sample fitted values already carry information from the actual mean0 of the same samples, so it overfits. The out-of-fold version is the honest comparison, and at lambda 10 it gives 0.9795, which is no real gain. Sentence-level regression on only 1496 samples is also weak, with R² around 0.54.

Token-level regression should do much better here, since it gives far more training points. The open question is whether mean0 information is even recoverable from layer 11 token states; token identity should be fairly easy to recover there, though mean0 also includes a length-dependent position component, which is minor.

I'm weighing whether to deliver mean0 to the final layer some other way, since there's no easy token-level reconstruction from h11 without the layer-12 head. Routing it through the SEP token won't help because every other token attends to it too, and nudging the CLS state at layer 0 runs into the full-rank K and V matrices and the LayerNorm and FFN after them. So I'm considering whether cutting the network at a different layer, using layer 10's output instead, would be a better place to work from.

I'm checking whether dropping mean0 into CLS at layer 0 could work if I also stop the other tokens from shifting when they attend to it. I'm also comparing how much the mean4 and mean9 variants gain against the base, to see whether cutting the network at a different layer would help at all.

I'm weighing whether mean0 could reach the CLS token at layer 10 instead, but it isn't available there either, only through the residual stream. Another option would be to make CLS invisible to the other tokens so its carried information stays put, though that would redirect their attention elsewhere and shift things anyway. Since the final readout gets refit regardless, the layer-0 trick already showed no gain (0.975 against 0.949), so I'm considering whether lexical features might do better instead.

I'm checking whether the gain from mean0 holds up, so I'll test more seeds and tune lambda and the scale of the mean0 block per combination. I'm also wondering whether an earlier layer's mean could be decoded more easily, though the difficulty looks the same. Then I'll decide whether this is worth pursuing.

I'm working through whether the LayerNorm gain and bias could be set so a layer passes its input through almost unchanged, but a normalized vector only passes through exactly if the gain and bias match each token's scale, which varies, so it wouldn't be a true identity.

I'm considering whether a low-dimensional projection of mean0 could be carried alongside mean11, since both are 768-dimensional and can't be stored losslessly in the residual stream together.

I'm trying to get mean0 information into the CLS token at layer 0 without disturbing the other tokens, since they attend to CLS from layer 1 onward. That only works if CLS's keys and values stay the same across layers 1 through 10, which is hard to guarantee.

I'm weighing whether mean0 could reach CLS at layer 0 without disturbing the other tokens, but their keys and values read CLS from layer 1 onward, so that shift is hard to avoid. The same problem would apply to mean4.

I'm now considering RBF kernels with more centers, though the earlier scans all landed in the 0.978–0.984 range, which looks like noise. A bigger lever might be more training data: hand-written sentences that mimic the practice style and target the confusable label pairs, such as premium versus term_cancellation, deductible versus premium, coverage versus deductible, and claims versus conditions. Some of that ambiguity is inherent, so the label conventions in the practice set will matter most.

I'm now asking whether a kernel machine built on all the practice sentences as centers holds up, and it looks robust. A separate idea is to combine the pooled layer-11 output with the mean0 features, though I can't keep both the original layer-11 output and my own pooling unless I find a way to carry them together.

I'm weighing how pooling inside the final layer could work: with 12 heads of arbitrary attention, the CLS output could only approximate mean11, since the original layer 11 attention can't run afterward to produce a true mean12.

I'm weighing whether pooling could carry more than a plain mean, by splitting the 12 heads so some compute a uniform mean of a reduced projection of layer 11 and others use attention weights that favor class-discriminative directions. I'm also considering down-weighting special tokens, punctuation, and stop words so the pooled vector leans on content words.

I'm weighing a pooling scheme where some attention heads take a uniform mean of a reduced projection of layer 11 while others favor class-discriminative directions. I'm also thinking about how to avoid the 64-texts-per-pass limit by designing the attention patterns from principles instead of token-level data, and then reading pooled features from the edited model in two passes.

I'm weighing a probe design for layer 11, where different attention heads carry different poolings in separate subspaces, with the FFN turned off. The final LayerNorm complicates reading the variants back out, since it normalizes the whole CLS vector and only approximately undoes the mixing, so I'd rather read each variant directly.

I'm considering whether the layer 12 output could be read pre-LayerNorm: with the FFN zeroed, the final vector is just the normalized attention mix plus the CLS residual, so the pooled part could be recovered up to two unknown scalars.

I'm settling on a cleaner way to get the pooled features: the attention output can be made to dominate the residual so the original CLS contribution is negligible. The layer-11 attention would then give a normalized mix of its own, and since the final design uses the same attention and LayerNorm with the FFN on top, the probe matches what I'd actually deploy.

I'm weighing whether the values could carry a reduced projection of the layer-11 tokens, since the full 768 dimensions don't have to be copied one-for-one. Each head's 64-dimensional value could come from a different linear map, so the pooled variants would share the attention output without needing a separate pass for each.

I'm checking whether the 768-dimensional mean can be squeezed into 384 PCA dimensions without losing much kernel accuracy, which would free six heads for other poolings. I'd test k at 384, 256, and 192 against the 0.98 baseline, and then consider content-weighted or max-style pooling in the remaining heads.

I'm considering whether each head could run its own softmax pooling by using a constant query and a linear score along a chosen direction, so tokens that point strongly along that direction get weighted more. The candidate directions would come from class-discriminative LDA on the layer-11 means, and the values could be 64-dimensional top components or LDA projections, with twelve heads to divide among the variants.

I'm weighing whether to free up some heads by reducing the mean-11 features to fewer PCA dimensions, and whether the remaining heads could carry class-specific pooling. The gain is uncertain, and the temperature for softmax pooling would be guesswork without token-level data. Before committing, I want to check kernel accuracy on PCA-reduced features and whether combining cls11 with mean11 helps.

I'm checking whether PCA-reducing the mean-11 features works, and whether combining cls11 with mean11 under different weightings is actually optimal.



Sentence-level data alone gives no honest gain (0.9795 out-of-fold, no better than baseline), so I'm turning to the 12 uniform heads of layer 11 and checking how many PCA dimensions the kernel needs.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=len(y)\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\ndef stdz(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef cvk(Z, lam=0.01, beta=1.0, seeds=3, D=None):\n    D=D or Z.shape[1]\n    K=gelu(beta*Z@Z.T/D); accs=[]\n    for seed in range(seeds):\n        f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            tr=f!=j; te=~tr; T=np.eye(8)[y[tr]]; b=T.mean(0)\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T-b); P[te]=K[np.ix_(te,tr)]@A+b\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nZ0=stdz(ln(d['cls11']+2*d['mean11']))\nU,S,Vt=np.linalg.svd(Z0,full_matrices=False)\nev=S**2/np.sum(S**2); print('cum var', [round(ev[:k].sum(),3) for k in [64,128,256,384,512]])\nfor kk in [128,256,384,512]:\n    Zk=Z0@Vt[:kk].T\n    print(kk, round(cvk(Zk,D=768),4))\n"
}
```


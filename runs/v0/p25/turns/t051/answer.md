## reasoning summary

I'm weighing whether the remaining errors are just ambiguous sentences, since the leftover rate sits near the label-noise floor, so richer representations might only give small gains. One option I'm considering is ensembling the kernel with extra feature sets from layer 11, such as non-uniform attention patterns, though that's hard to evaluate without token-level hidden states. I was also starting on a second option, but it trails off there.

Next I'm checking robustness, since the kernel uses 1496 neurons with large output weights (S=100). The FFN writes 8 scaled scores into the lowest-variance directions of the LN1 output, and LN2 then normalizes over a vector dominated by that term. The scale is common across all 8 dimensions, so the argmax should be unaffected, but I still need to see how the pooler reads those directions.

I'm checking whether the small leftover noise in U^T n could flip an argmax when the class score differences are tiny, and estimating its scale against S·F; it looks negligible. I'm also looking at tanh saturation at the classifier output, where the top-class logit reaches about 8.68.

I'm checking tanh saturation in the classifier output, which looks moderate but stays monotone, so the argmax holds. The main open question is whether to chase the marginal ~0.3% gain from adding mean0, which would mean routing bag-of-embeddings information into layer 11's CLS without disturbing the other features.

I'm wondering whether layer 11 could supply extra information that the current features don't capture. Using a different attention pattern, like a mean over tokens that excludes CLS and SEP, might add something, but the cls12 info I tested earlier was weaker. I think the realistic gain here is small, though the kernel's CV around 0.981 leaves some room to question it.

I'm considering whether to augment the kernel's training set with my own labeled insurance sentences, since 1576 center slots remain unused beyond the 1496 already taken. Writing a few hundred to a thousand new examples and reading their features in one pass could help generalization.

I'm checking whether adding my own labeled insurance sentences to the kernel's training set would help. First I want the learning curve: if accuracy keeps rising as the training fraction grows, more data should help. I'd also measure whether CV on the practice folds improves when my sentences are always included in training, since that's the real test of whether they help.

I'm weighing whether to hand-write a few hundred extra labeled insurance sentences to add to the kernel's training set, since that would be roughly 50 per class. Templated versions would likely be too repetitive to help, so any gain depends on the hand-written ones being diverse. Before committing to that effort, I want the learning curve to show whether accuracy is still rising with more training data.



Most remaining errors are truly ambiguous sentences, so the 1496 practice sentences are near the noise floor. I'll check whether the learning curve is still rising, since adding more labeled data from sentences I write could help.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\nX=LN(f['cls11']+2*f['mean11']); mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd; d=768\ndef run(frac, seed, lam=0.01):\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%5\n    acc=[]\n    for k in range(5):\n        tr=np.where(fo!=k)[0]; te=fo==k\n        tr=rng.permutation(tr)[:int(frac*len(tr))]\n        P=gelu(Z[tr]@Z[tr].T/d); T=np.eye(8)[y[tr]]; Pm=P.mean(0); Tm=T.mean(0); Pc=P-Pm\n        A=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n        acc.append((np.argmax((gelu(Z[te]@Z[tr].T/d)-Pm)@A+Tm,1)==y[te]).mean())\n    return np.mean(acc)\nfor frac in [0.25,0.5,0.75,1.0]:\n    print(frac, int(frac*1197), round(np.mean([run(frac,s) for s in range(6)]),4))\n"
}
```


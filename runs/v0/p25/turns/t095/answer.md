## reasoning summary

I'm checking the 128-dimension result, which dropped to 0.596, and suspect the kernel scale is off at D=768. At 384 dimensions I get 0.9777 against 0.980 before, so there's a small loss, and beta might need retuning. The freed-heads idea is speculative, so I'm looking for something with a more reliable gain, and the "mean0" route is next.

I'm weighing whether the mean0 information could be carried by a token other than CLS, since CLS is heavily attended in every layer. One option is to push the mean into CLS only at a late layer, around 10 to 11, where it would affect fewer downstream reads. The mean at layer 4 and layer 9 already gives small gains, so a late-layer variant might be worth trying.

I'm looking at whether a late-layer variant could work: at layer 10, CLS could pool over the layer-10 tokens with all 12 heads and run its own FFN, changing only its own output, so the other tokens are undisturbed. Layer 11 would then pool again over the layer-11 tokens, giving two pooling stages. The open question is whether the CLS row at layer 10 is distinctive enough to query reliably.

Since mean10 adds nothing, I'm dropping that branch; the useful signal came from mean0 and mean4, which the later layers' contextualized averages wash out. I'm also wondering whether the layer-10 CLS update would shift the other tokens' means enough to matter, and I'm not sure yet.

I'm checking whether a kernel CV on the l0trick features, the [cls11', mean11'] pair, would show anything, since the mean11' shift was large. It's a quick and cheap test against the saved data.



Compressing mean11 to 384 dimensions loses a bit (0.9777 vs 0.980), so I'm re-testing my layer-0 CLS-trick features with the kernel next, since they've only been checked with linear classifiers so far.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=len(y)\nt=np.load('l0trick_feats.npz'); print({k:t[k].shape for k in t.files})\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\ndef stdz(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef cvk(Z, lam=0.01, beta=1.0, seeds=3):\n    K=gelu(beta*Z@Z.T/Z.shape[1]); accs=[]\n    for seed in range(seeds):\n        f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            tr=f!=j; te=~tr; T=np.eye(8)[y[tr]]; b=T.mean(0)\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T-b); P[te]=K[np.ix_(te,tr)]@A+b\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nks=[k for k in t.files]\nc=[k for k in ks if 'cls' in k and '11' in k]; m=[k for k in ks if 'mean' in k and '11' in k]; print(c,m)\nif c and m:\n    for a in [1,2,4]:\n        print(a, round(cvk(stdz(ln(t[c[0]]+a*t[m[0]]))),4), round(cvk(np.hstack([stdz(t[c[0]]),stdz(t[m[0]])])),4))\n"
}
```


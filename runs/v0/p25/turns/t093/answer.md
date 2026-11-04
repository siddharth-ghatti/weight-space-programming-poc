## reasoning summary

Adding mean0 nudged the score up about half a point, from 0.980 to 0.985. Since mean0 is just the bag of embedding-layer tokens, that gain is probably lexical signal. I'm not yet sure a half-point difference is real rather than seed noise, so I'm checking that next.

I'm weighing how to get the mean0 lexical signal into the final CLS without disturbing the other tokens. The layer-0 CLS trick failed because it shifted the attention-sink tokens, and I don't see an unused subspace to hide it in. So I'm considering whether mean11 already captures enough of that token identity that a kernel on mean0 could be approximated from the layer-11 token states instead.

I'm noticing that mean0 is a linear function of the token states, so if token identity is decodable from layer 11, mean0 should be roughly recoverable from mean11 via a linear map. That means the mean0 signal might be reachable without a separate read. The open question is whether a linear transform of mean11 can stand in for the mean0 kernel feature inside the existing layer-11 design.

I'm weighing whether a linear map from mean11 could stand in for mean0, since mean0 is itself linear in the token states. If it could, the kernel's lexical feature might fit inside the existing layer-11 design without a separate read. Next I want to check how well mean11 actually predicts mean0.

I'm testing whether a ridge map from mean11 can predict mean0, using a kernel on the resulting features. The map doesn't use labels, so fitting it on all samples is fine. If the token-level recovery is weak, I could read token hidden states for a few texts at layers 0 and 11 and fit the map there.

I'm checking whether a ridge map from mean11 can recover mean0 well enough, and I'm planning to read token-level hidden states at layers 0 and 11 for a few texts if the sentence-level test is too weak.



Adding `mean0` lifts kernel CV from about 0.980 to 0.985, and since token identity looks linearly readable from layer 11, I'm testing whether a ridge map from `mean11` to `mean0` can stand in for it.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=len(y)\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\ndef stdz(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef cvk(Z, lam=0.01, beta=1.0, seeds=3):\n    K=gelu(beta*Z@Z.T/Z.shape[1]); accs=[]\n    for seed in range(seeds):\n        f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            tr=f!=j; te=~tr; T=np.eye(8)[y[tr]]; b=T.mean(0)\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T-b); P[te]=K[np.ix_(te,tr)]@A+b\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nZ0=stdz(ln(d['cls11']+2*d['mean11']))\nX=np.hstack([d['cls11'],d['mean11']]); Xc=X-X.mean(0); Y=d['mean0']-d['mean0'].mean(0)\nfor lam in [1,10,100]:\n    # out-of-fold R2 of regression (unsupervised)\n    f=np.arange(n)%5; Yh=np.zeros_like(Y)\n    for j in range(5):\n        tr=f!=j; B=np.linalg.solve(Xc[tr].T@Xc[tr]+lam*np.eye(X.shape[1]),Xc[tr].T@Y[tr]); Yh[~tr]=Xc[~tr]@B\n    B=np.linalg.solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@Y); Yfit=Xc@B\n    print(lam, 'oofR2', 1-((Y-Yh)**2).sum()/(Y**2).sum(), 'cv kernel w/ oof mean0hat', round(cvk(np.hstack([Z0,stdz(Yh)])),4), 'w/ fit', round(cvk(np.hstack([Z0,stdz(Yfit)])),4))\n"
}
```


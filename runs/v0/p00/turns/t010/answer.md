## reasoning summary

I'm checking whether the self-response stays high while the cross-response has a median near 0.48, which suggests related tokens like "cover" and "covers" are driving that overlap. Setting the threshold around 0.72 leaves a self margin of about 0.21, but the 99th percentile of cross-response sits at 0.76, which is above threshold, so that's a problem.

I'm checking which tokens produce those high cross-responses, since the median of 0.48 is too high for random noise in 768 dimensions and suggests real near-neighbors like morphological variants. I'm also considering whether the least-squares FFN fit, rather than a hard threshold, could handle this crosstalk in a useful way for unseen tokens.



Self-responses stay very stable across positions (≥0.93), but a typical detector has another token reaching about 0.48. I'm checking which tokens those are, since morphological variants are harmless but large-norm junk tokens would need handling.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nCb=np.load('Cb.npy'); Cw=np.load('Cw.npy'); mu=np.load('mu.npy'); K=np.load('K.npy')\ninv = {i:t for t,i in vocab().items()}\nallv=np.arange(30522)\nMk = sum(em.h0(K,p) for p in [3,15,35])/3\nC = Cb + 3*Cw + 0.05*np.eye(768)\nU = np.linalg.solve(C, (Mk-mu).T).astype(np.float32)\nU /= np.einsum('dk,kd->k',U,(Mk-mu))[None,:]\nH20 = em.h0(allv,20)-mu\nR = H20@U[:, :400]  # top-400 frequent tokens\nfor i in list(range(0,400,25)):\n    r = R[:,i].copy(); r[K[i]]=-9\n    top = np.argsort(-r)[:5]\n    print(inv[K[i]], [(inv[t], round(float(r[t]),2)) for t in top])\n# whitened norms\nCinv_h = np.linalg.solve(C, H20.T)\nwn = np.einsum('dj,jd->j', Cinv_h, H20)\nprint('whitened sqnorm percentiles', np.percentile(wn,[1,10,50,90,99]))\nprint('K whitened sqnorm percentiles', np.percentile(wn[K],[1,10,50,90,99]))"
}
```


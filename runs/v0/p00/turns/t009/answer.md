## reasoning summary

The covariance matrix is rank-deficient, so I need to regularize it before inverting. Next I'll set up the detector vectors and check responses at a few token positions.

I'm checking how the detector responses scale over the vocabulary, since the 30522 × 3411 matrix is around 400MB per position and may need to be chunked. I also want the max cross-response for each detector and a count of detectors firing above 0.5 per vocab token. Then I'll try a few regularization settings and compare the self-response and cross-max distributions.



The covariances are rank-deficient, so I'm adding regularization and then comparing whitening settings by how selectively each detector responds.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nCb=np.load('Cb.npy'); Cw=np.load('Cw.npy'); mu=np.load('mu.npy'); K=np.load('K.npy')\nallv=np.arange(30522)\nmpos=[3,15,35]\nMk = sum(em.h0(K,p) for p in mpos)/3\nselfpos = [1,2,4,8,16,32,50,64,80]\nHk = np.stack([em.h0(K,p) for p in selfpos],1)  # [K,np,768]\nHall = {p: em.h0(allv,p)-mu for p in [2,20,50]}\nfor a,lam in [(1,0.01),(3,0.01),(3,0.05),(10,0.05),(3,0.2)]:\n    C = Cb + a*Cw + lam*np.eye(768)\n    U = np.linalg.solve(C, (Mk-mu).T).astype(np.float32)  # [768,K]\n    U /= np.einsum('dk,kd->k',U,(Mk-mu))[None,:]\n    selfr = np.einsum('kpd,dk->kp', Hk-mu, U)\n    crossmax = np.full(len(K), -9.0)\n    for p,Hp in Hall.items():\n        for s in range(0,30522,5000):\n            R = Hp[s:s+5000]@U  # [chunk,K]\n            # mask self\n            idx = np.where((K>=s)&(K<s+5000))[0]\n            R[K[idx]-s, idx] = -9\n            crossmax = np.maximum(crossmax, R.max(0))\n    print(f'a={a} lam={lam} self: min {selfr.min():.3f} p1-mean {selfr[:,0].mean():.3f} p80-mean {selfr[:,-1].mean():.3f} 1pct {np.percentile(selfr.min(1),1):.3f} | cross max: median {np.median(crossmax):.3f} 90% {np.percentile(crossmax,90):.3f} 99% {np.percentile(crossmax,99):.3f} max {crossmax.max():.3f}')"
}
```


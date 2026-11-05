## reasoning summary

The simulation is noisy because there are only 1000 base items and flips are rare, with m1792's RMS error at 0.139 landing in between. The estimated risk hangs on a handful of specific flipped items, so the sampling noise is large. A smoother option is to treat the noise as random with an item-specific scale, drawing each perturbation from a normal distribution whose spread varies per item.

I'm sizing up the noise model: each item gets its own Gaussian perturbation, scaled by its average sensitivity across the three models and normalized to the model's overall RMS. I'll draw many noise realizations per item, then trace P(fail) against RMS for each k and map the three model sizes onto their RMS values to decide with expected value.



Direct simulation is too noisy, since only about 20 in 1000 items flip, so I'll smooth it by modeling each item's logit noise as Gaussian with a per-item sensitivity scaled by a global level. That lets me map fail probability against rms noise for each k.



## tool_use: run_python

```json
{
 "code": "import json\nexec(open('risk.py').read())\nM={'m1536v1':np.load('lg_pass25.npy'),'m1792v2':np.load('lg_pass31_m1792v2.npy'),'m2048v2':np.load('lg_pass30_m2048v2.npy')}\nS=[]\nfor n,l in M.items():\n    r=np.sqrt(((l-bl)**2).mean()); S.append(np.sqrt(((l-bl)**2).mean(1))/r)\nsens=np.sqrt(np.mean(np.array(S)**2,0)); sens/=np.sqrt((sens**2).mean())\ndef risk_g(rms, k, R=20000, N=400, seed=2, allow=4):\n    rng=np.random.default_rng(seed)\n    P=sm(bl/k); cum=P.cumsum(1)\n    idxs=rng.integers(0,1000,(R,N)); u=rng.random((R,N))\n    G=(u[...,None]>cum[idxs]).sum(-1)\n    bp=bl.argmax(1)[idxs]; fb=f1macro_batch(G,bp); ab=(bp==G).sum(1)\n    noise=rng.standard_normal((R,N,4))*(rms*sens[idxs])[...,None]\n    ep=(bl[idxs]/k+noise).argmax(-1)\n    da=(ep==G).sum(1)-ab; fe=f1macro_batch(G,ep)\n    fa=da<-allow; ff=fe<fb-0.02\n    return (ep!=bp).sum(1).mean(), da.mean(), fa.mean(), ff.mean(), (fa|ff).mean()\nfor k in [1.0,1.25,1.5]:\n    for r in [0.0,0.08,0.106,0.139,0.172,0.22,0.30]:\n        print('k=%.2f rms=%.3f flips %5.1f E[da] %5.2f P(acc) %.3f P(f1) %.3f P(fail) %.3f'%((k,r)+risk_g(r,k)))\n"
}
```


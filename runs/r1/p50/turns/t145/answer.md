## reasoning summary

Low-margin items show much larger noise, with their sensitivity tracking how close they sit to the decision boundary. That would explain why the empirical simulation produces more flips than the Gaussian model. I'm checking whether the per-item sensitivity, computed from the RMS across all four logits, already accounts for this.

I'm working out whether the noise along the margin direction explains the gap: if the perturbation is aligned with the top-two logit difference, the per-logit RMS understates the change in margin by about a factor of two. That would mean the Gaussian model underestimates the flips, and the empirical simulation is closer to right. I also want to check whether the memorization component on base-right low-margin items overstates the risk on test data.

I'm considering whether to symmetrize the simulation by applying the noise with random sign, so the systematic shift from memorization doesn't bias the flip count. I'm also weighing a margin-aware alternative where each item's noise is borrowed from a comparable-margin item, though that breaks alignment with the top classes unless the noise is re-expressed relative to the new item's own top-two logits.

I'm now planning to match each synthetic item's noise to a donor item with a similar actual base margin, and to express that noise in the donor's rank order so it lines up with the new item's top classes. I'm also weighing whether to randomize the sign so the systematic memorization shift doesn't bias the flip count.

I'm weighing two options for the synthetic noise: keep the sign, which retains the memorization shrink, or randomize it. Either way, I'll match each item to about 30 donors with similar margin so the estimate is smoother.



Isotropic noise underestimates flips, since noise is much larger for low-margin items and lies along the decision direction. Next I'm running a more principled simulation that uses shrunk base logits and donor noise, with both the original and a random sign.



## tool_use: run_python

```json
{
 "code": "import json\nexec(open('risk.py').read())\ns_=np.sort(bl,1); margA=s_[:,-1]-s_[:,-2]; rank=np.argsort(-bl,1)  # rank[i,r]=class at rank r\ndef rankframe(lg):\n    d=lg-bl; return np.take_along_axis(d,rank,1)   # noise ordered by base rank\norder=np.argsort(margA)\ndef risk_donor(lg, k, R=6000, N=400, seed=3, allow=4, randsign=False, nd=30):\n    rng=np.random.default_rng(seed)\n    D=rankframe(lg)\n    P=sm(bl/k); cum=P.cumsum(1)\n    idxs=rng.integers(0,1000,(R,N)); u=rng.random((R,N))\n    G=(u[...,None]>cum[idxs]).sum(-1)\n    bp=bl.argmax(1)[idxs]; fb=f1macro_batch(G,bp); ab=(bp==G).sum(1)\n    # donor: item with margin closest to margA[j]/k, random among nd nearest\n    pos=np.searchsorted(margA[order], margA/k)          # per item j\n    off=rng.integers(-nd//2, nd//2, (R,N))\n    dpos=np.clip(pos[idxs]+off,0,999); donor=order[dpos]\n    dn=D[donor]*(np.where(rng.random((R,N))<.5,-1,1)[...,None] if randsign else 1)\n    noise=np.zeros_like(dn); \n    np.put_along_axis(noise, rank[idxs], dn, -1)        # back to class frame of item j\n    ep=(bl[idxs]/k+noise).argmax(-1)\n    da=(ep==G).sum(1)-ab; fe=f1macro_batch(G,ep)\n    fa=da<-allow; ff=fe<fb-0.02\n    return (ep!=bp).sum(1).mean(), da.mean(), fa.mean(), ff.mean(), (fa|ff).mean()\nM={'m1536v1':np.load('lg_pass25.npy'),'m1792v2':np.load('lg_pass31_m1792v2.npy'),'m2048v2':np.load('lg_pass30_m2048v2.npy'),'m1536v3':np.load('lg_pass29.npy')}\nfor rs in [False,True]:\n  for k in [1.0,1.25,1.5]:\n    for n,l in M.items():\n        print('rs=%d k=%.2f %-8s flips %5.1f E[da] %5.2f P(acc) %.3f P(f1) %.3f P(fail) %.3f'%((rs,k,n)+risk_donor(l,k,randsign=rs)))\n"
}
```


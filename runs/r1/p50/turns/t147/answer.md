## reasoning summary

I'm noting that the three variants are statistically indistinguishable, so there's nothing to choose between them. Given the 19 passes and 178 code runs, I'm asking whether any strategy is left that could meaningfully cut the noise at m1536. I'm also weighing the idea that the noise is roughly a sum of independent per-layer contributions, each around 20%, so any gain would have to come from the individual layers.

I'm now checking whether the compensation could go beyond the kept units' activations, since the residual error might be partly linear in the FFN input itself. I'm also wondering whether the next layer's attention could absorb a linear piece of that error, though that seems hard to do directly.

I'm trying to see whether the missing contribution could be absorbed one layer later, since the next layer's input is a normalized version of the same signal. The catch is that the correction would depend on the pre-normalized FFN input, which isn't directly available to the next layer, so a linear fit onto that input only approximates it.

I'm checking whether the residual could be absorbed linearly into the next layer's attention inputs, since they see a normalized version of the signal, but the residual connection path doesn't have a matching term to absorb it into. That leaves the correction only partly achievable, and it wouldn't be exact.

I'm considering whether the reduced-rank linear mapping already tried in layer 5 could be revisited, since it suggested the residual isn't very predictable from the FFN input. I'm also thinking about whether the CLS-only layer-5 FFN could be repurposed to correct upstream error, though the earlier end-to-end attempt may simply have lacked enough data to show a benefit.

I'm now looking at whether the layer-5 attention output could carry any of the error, since the CLS row there is what actually matters. That error would come from upstream token values and keys plus the CLS query, which feels like a dead end for a direct fix.

I'm weighing whether to keep collecting more data at m1536 with the remaining passes, since the gains from a larger sample look small, around 2%. I'm also deciding between v1 and v2, though the naming around lg_m1536_v2.npy looks confused, since its numbers match pass25 rather than pass26.

I'm weighing whether the remaining passes are worth spending on a final check: the sensible move is to restore the v1 setup and run one predict pass to confirm it reproduces the earlier agreement figure. I'm also reconsidering m1280, since its expected value came out level with m1536, and I'm checking whether my risk estimates are too pessimistic given the random-sign donor simulation.

I'm weighing whether to finish with v1 or v2, and v2 has slightly better agreement and RMS but a slightly lower DM overall. I'm planning to restore v1, rerun a single predict pass to confirm it reproduces the earlier numbers, and then pick between them from that.

I'm going to pick between v1, v2 and v3 using a smoothed metric rather than the single noisy pass numbers, so I'll run the donor simulation with more repetitions and both signs, and compare them that way.



v1, v2 and v3 look identical on the low-margin noise (0.51–0.52), and v3 beats v1 only 45% of the time, so I'm rerunning the smoothed donor simulation on all three at k=1.25 with both sign variants and more repetitions to separate them.



## tool_use: run_python

```json
{
 "code": "import json\nexec(open('risk.py').read())\ns_=np.sort(bl,1); margA=s_[:,-1]-s_[:,-2]; rank=np.argsort(-bl,1); order=np.argsort(margA)\ndef risk_donor(lg, k, R=6000, N=400, seed=3, allow=4, randsign=False, nd=30):\n    rng=np.random.default_rng(seed)\n    D=np.take_along_axis(lg-bl,rank,1)\n    P=sm(bl/k); cum=P.cumsum(1)\n    idxs=rng.integers(0,1000,(R,N)); u=rng.random((R,N))\n    G=(u[...,None]>cum[idxs]).sum(-1)\n    bp=bl.argmax(1)[idxs]; fb=f1macro_batch(G,bp); ab=(bp==G).sum(1)\n    pos=np.searchsorted(margA[order], margA/k); off=rng.integers(-nd//2, nd//2, (R,N))\n    donor=order[np.clip(pos[idxs]+off,0,999)]\n    dn=D[donor]*(np.where(rng.random((R,N))<.5,-1,1)[...,None] if randsign else 1)\n    noise=np.zeros_like(dn); np.put_along_axis(noise, rank[idxs], dn, -1)\n    ep=(bl[idxs]/k+noise).argmax(-1)\n    da=(ep==G).sum(1)-ab; fe=f1macro_batch(G,ep)\n    return ((da<-allow)|(fe<fb-0.02)).mean()\nM={'v1':np.load('lg_pass25.npy'),'v2':np.load('lg_pass26.npy'),'v3':np.load('lg_pass29.npy')}\nfor n,l in M.items():\n    r=[risk_donor(l,1.25,seed=s,randsign=rs) for s in (11,12) for rs in (False,True)]\n    print(n, np.round(r,3), 'mean %.3f'%np.mean(r))\n"
}
```


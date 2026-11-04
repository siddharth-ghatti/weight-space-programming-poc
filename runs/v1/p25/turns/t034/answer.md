## reasoning summary

Now I'm comparing the results across the base and tapped variants, with the seed7 folds for reference. The gaps are around 0.3 to 0.5 percent, which is small enough that I'm treating it as noise, though I want to see how the last comparison ends.

I'm going to set up a repeated cross-validation over five seeds with five folds each, covering five candidate feature sets, since the single-run gaps are too close to call. Using the dual kernel solve should keep that tractable.



Readout at R=9 or 10 looks about 0.4–0.6% better than R=11, and layer-6-tapped [C10′, M10′] scores best at about 97.5%, though those gaps may be CV noise. I'm running repeated 5×5 CV on the candidates to decide.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; A=dict(np.load('featsA.npz'))\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef mkfolds(seed):\n    rng=np.random.RandomState(seed); f2=np.zeros(len(y),int)\n    for c in range(8):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f2[idx]=np.arange(len(idx))%5\n    return f2\nFS=[mkfolds(s) for s in [0,7,11,23,99]]\ndef rcv(C,M,wc,lams):\n    out=np.zeros(len(lams))\n    for fo in FS:\n        for f in range(5):\n            tr=fo!=f; te=~tr\n            Z=np.concatenate([wc*stdz(C,tr),stdz(M,tr)],1); Z=Z-Z[tr].mean(0)\n            K=Z@Z[tr].T; Y=np.eye(8)[y[tr]]-1/8\n            ev,U=np.linalg.eigh(K[tr]); UtY=U.T@Y\n            for i,l in enumerate(lams):\n                a=U@(UtY/(ev+l)[:,None]); out[i]+=(np.argmax(K[te]@a,1)==y[te]).sum()\n    return np.round(out/(len(FS)*len(y)),4)\nlams=[150,300,500,800]\ncands={'base9':(F['C9'],F['M9']),'base10':(F['C10'],F['M10']),'base11':(F['C11'],F['M11']),'tap9':(A['C9'],A['M9']),'tap10':(A['C10'],A['M10']),'tap11':(A['C11'],A['M11'])}\nfor k,(C,M) in cands.items():\n    for wc in [0.7,1.0,1.3]:\n        print(k,wc,rcv(C,M,wc,lams))"
}
```


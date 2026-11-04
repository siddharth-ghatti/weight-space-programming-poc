## reasoning summary

Moving the readout to layer 9 or 10 looks worth about 0.4% over layer 11, with accuracy staying around 97%. Since layers after the readout become identity, there's no perturbation cost. So at R=9, layer 9 handles the readout by pooling the CLS token's values through head 0 with the residual from C9, and layers 10 and 11 pass through unchanged.

Checking the identity layers now: with the FFN zeroed and attention output zero, layers 10 and 11 just renormalize an already normalized vector, so the CLS state passes through them unchanged. The pooler stays the same as before, applied to the layer-9 readout.

I'm checking whether tapping the CLS token at an earlier layer, around 6 or 7, would carry more of layer 6's information into the layer-9 readout, and I'll compute those combinations from the featsA features.



Moving the readout to layer 9 costs nothing, with [C9, M9] around 97.0–97.25% against 96.7% for layer 11. I'm checking whether the layer-6 tap adds anything at R=9/10 next.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy'); A=dict(np.load('featsA.npz'))\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam, folds=folds):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor R in [9,10]:\n    for wc in [0.7,1.0]:\n        fn=lambda tr: np.concatenate([wc*stdz(A[f'C{R}'],tr),stdz(A[f'M{R}'],tr)],1)\n        print(\"tapped [C%d',M%d'] wc\"%(R,R),wc,[cv(fn,l) for l in [100,200,300,500,800]])\n        fn=lambda tr: np.concatenate([wc*stdz(F[f'C{R}'],tr),stdz(F[f'M{R}'],tr)],1)\n        print(\"base   [C%d,M%d] wc\"%(R,R),wc,[cv(fn,l) for l in [100,200,300,500,800]])\n# robustness: different fold seeds for base C9+M9 vs C11+M11\nrng=np.random.RandomState(7); f2=np.zeros(len(y),int)\nfor c in range(8):\n    idx=np.where(y==c)[0]; rng.shuffle(idx); f2[idx]=np.arange(len(idx))%5\nfor R in [9,10,11]:\n    fn=lambda tr: np.concatenate([0.85*stdz(F[f'C{R}'],tr),stdz(F[f'M{R}'],tr)],1)\n    print('seed7 base [C%d,M%d]'%(R,R),[cv(fn,l,f2) for l in [200,300,500]])"
}
```


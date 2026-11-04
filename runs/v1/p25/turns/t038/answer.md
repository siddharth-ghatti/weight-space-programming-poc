## reasoning summary

I've got the layer-9 readout with identity layers 10 and 11 confirmed as exact, with expected test accuracy around 97.5%. With the remaining 15 passes, I'm weighing whether a tap into CLS from an earlier layer, around 4 or 5, could feed a readout at layers 6 or 7 and do better than the earlier tap at layer 6, which didn't help.

I'm weighing a few more directions for the remaining passes: whether a tap at layer 3 or 4 could feed a readout at layer 6, since the earlier tap didn't help. I'm also considering whether the free identity layers 10 and 11 could add anything, though the tokens there only carry what layer 9 already produced, so probably not.

I'm considering whether a nonlinear readout could do better, since the free layers 10 and 11 can't add new information on their own. The question is whether features derived from the shallow tokens, like M0 or M4 together with C9 and M9, would give a meaningful gain, and that's worth checking offline before putting more effort into it.

I'm thinking about whether a kernel method, like RBF kernel ridge on [C9, M9], could beat the linear readout in cross-validation, and whether reaching around 98% would make it worth pursuing.



The layer-9 readout matches the 97.46%-CV classifier, so the model now encodes it. With 15 passes left, I'll check offline whether adding shallow mean features or an RBF readout on [C9, M9] is worth the work before spending any passes.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef mkfolds(seed):\n    rng=np.random.RandomState(seed); f2=np.zeros(len(y),int)\n    for c in range(8):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f2[idx]=np.arange(len(idx))%5\n    return f2\nFS=[mkfolds(s) for s in [0,7,11]]\ndef rcvZ(build,lams,kern=None):\n    out=np.zeros(len(lams))\n    for fo in FS:\n        for f in range(5):\n            tr=fo!=f; te=~tr\n            Z=build(tr); Z=Z-Z[tr].mean(0)\n            if kern is None: K=Z@Z[tr].T\n            else:\n                sq=(Z**2).sum(1); D=sq[:,None]+sq[tr][None,:]-2*Z@Z[tr].T; K=np.exp(-kern*D/Z.shape[1])\n            Y=np.eye(8)[y[tr]]-1/8\n            ev,U=np.linalg.eigh(K[tr]); UtY=U.T@Y\n            for i,l in enumerate(lams):\n                a=U@(UtY/(ev+l)[:,None]); out[i]+=(np.argmax(K[te]@a,1)==y[te]).sum()\n    return np.round(out/(len(FS)*len(y)),4)\nprint('C9M9', rcvZ(lambda tr: np.concatenate([stdz(F['C9'],tr),stdz(F['M9'],tr)],1),[300,500,800]))\nfor extra in ['M0','M2','M4']:\n    print('C9M9+'+extra, rcvZ(lambda tr: np.concatenate([stdz(F['C9'],tr),stdz(F['M9'],tr),stdz(F[extra],tr)],1),[300,500,800,1200]))\nfor g in [0.1,0.3]:\n    print('rbf C9M9 g',g, rcvZ(lambda tr: np.concatenate([stdz(F['C9'],tr),stdz(F['M9'],tr)],1),[0.003,0.01,0.03,0.1],kern=g))"
}
```


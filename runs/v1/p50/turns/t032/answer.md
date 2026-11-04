## reasoning summary

I've got the per-layer scales back: about 4.72, 3.58, and 2.37 for layers 2, 6, and 10, with sentence-to-sentence variation around 9%, 7%, and 4%, and the residual is zero with per-layer scales. To even out the contribution, I'm thinking about scaling each gain by a reference value over its layer's scale, so every layer lands near the same average.

I'm weighing whether the per-sentence variation of about 9% in layer 2 will hurt generalization, so I'm planning to simulate out-of-fold predictions with per-sentence multiplicative noise to check. I'm also considering whether the LayerNorm factors causing that variation could be reduced by carrying the signal through fewer normalization steps.

I'm now planning to refit the ridge on features scaled by each sentence's actual LayerNorm factor, since the network computes those factors anyway and they don't depend on the chosen weights. That way the refit matches what the network really outputs, and I can evaluate out-of-fold with the same scaling in place.

I'm going to run out-of-fold predictions on both the a-scaled and unscaled features, then keep whichever generalizes better. The network computes the scaled version either way, so fitting on scaled features is the consistent choice.



Each carry layer has its own scale (layer 2 at 4.72, layer 6 at 3.58, layer 10 at 2.37), and with per-layer scales the decode residual drops to 0. Next I'll refit the joint ridge on the scaled features and check OOF accuracy against the unscaled version.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nF=dict(np.load(WORK/'prep_feats.npz')); A=np.load(WORK/'v2_layer_scales.npy'); abar=A.mean(0)\ndef oof_pred(X,lam):\n    o=np.zeros((N,8))\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X[tr]-mu)/sd\n        al=np.linalg.solve(Z@Z.T+lam*np.eye(tr.sum()),Y[tr]-Y[tr].mean(0)); o[te]=((X[te]-mu)/sd)@(Z.T@al)\n    return o\nacc=lambda o:round((o.argmax(1)==lab).mean(),4)\nXu=np.concatenate([F[p][:,KEEP] for p in ['m2','m6','m10','c12']],1)\nXs=np.concatenate([F[p][:,KEEP]*(A[:,k]/abar[k])[:,None] for k,p in enumerate(['m2','m6','m10'])]+[F['c12'][:,KEEP]],1)\nfor lam in [200,300,500]:\n    ou=oof_pred(Xu,lam); os_=oof_pred(Xs,lam)\n    # also: fit on unscaled, but evaluate with network's scaling (mismatch) -> emulate by fitting unscaled weights and applying to scaled features\n    print(lam,'unscaled-consistent',acc(ou),'scaled-consistent',acc(os_))\n# mismatch emulation: fit W on unscaled, apply to scaled features\nlam=300; o=np.zeros((N,8))\nfor f in range(5):\n    tr=folds!=f; te=~tr\n    mu=Xu[tr].mean(0); sd=Xu[tr].std(0)+1e-6; Z=(Xu[tr]-mu)/sd\n    W=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(Y[tr]-Y[tr].mean(0)))\n    o[te]=((Xs[te]-mu)/sd)@W\nprint('fit-unscaled, apply-scaled', acc(o))\n"
}
```


## reasoning summary

I've confirmed the real model's logits line up with the offline ones, and the edited model reproduces the stash readout exactly, so the honest CV sits around 98.0 to 98.1%. Next I'm weighing whether a kernel readout on top is worth the remaining passes, since it might add roughly 0.1% after the realizable losses.

I'm weighing whether the stash could carry a per-token linear write of the CLS state from an earlier layer, using the FFN in its near-linear GELU regime to copy a projection into the stash dims, and whether that could beat adding a kernel readout.

I'm checking whether the stash could also carry linear projections of earlier CLS states, using the near-linear GELU regime to write them into the stash dimensions, at the cost of a few FFN neurons per layer. Since the sum stays linear in logit contributions, the idea is simply to test whether adding C_l features beyond the current C9 and M9 with M0 gains anything, and whether reading out at layer 11 might work better than layer 9.

I'm checking whether the stash could also take linear write-ins of CLS states from later layers, like C11 and M11, alongside C9, M9 and M0. That would mean FFN linear units for CLS and uniform heads for the means. Next I want to test those combinations offline before deciding whether reading out at layer 11 is worth it.

I'm going to check offline whether adding C and M features from other layers improves on the current [C9, M9, M0] stash, especially a version that reads out at layer 11. The open question is whether a uniform head at layer 9 for M9 is worth the cost to the later layers. I'll start with the base features in feats.npz, which hold C0 through C12 and M0 through M12.



The real model matches the offline logits, so the stash and readout are correct and honest CV is about 98.0–98.1%. I'll note that, then check offline whether stashing more information from other layers adds a gain.



## tool_use: run_python

```json
{
 "code": "open('notes.md','a').write('''\n- RESULT stash (passes 11-14): stash exact (R2=1 at C9[S], alpha~.036, cv 9%). Perturbation cost ~0 (keep-only CV 97.8/97.5 vs base 97.6).\n  Honest CV (in-fold stash simulated via measured alpha/beta): ~98.0-98.1 vs 97.64 base. featsS.npz = C7-9',M7-9' (stash model).\n- SAVED: stash layers0-8 + L9 readout fit on C9'(incl S, ws=1, wc=.85) + M9'(nonS), lam500 (readout_S.npz). Verified pass 14: model logits == offline (corr>.9999).\n''')\nimport numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nG={k:F[k].astype(np.float64) for k in F.files if k!='y'}\nSz={k:(v-v.mean(0))/(v.std(0)+1e-9) for k,v in G.items()}\ndef cv_dual(keys,lams):\n    X=np.concatenate([Sz[k] for k in keys],1); K=X@X.T\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8; ym=Y.mean(0); n=tr.sum()\n            Ktr=K[np.ix_(tr,tr)]; H=np.eye(n)-1/n; Kc=H@Ktr@H\n            Kte=K[np.ix_(te,tr)]; Kte=Kte-Kte.mean(1,keepdims=True)-Ktr.mean(0)[None]+Ktr.mean()\n            for l in lams:\n                a=np.linalg.solve(Kc+l*np.eye(n),Y-ym); res[l]+=((Kte@a+ym).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nL=[500,1000,2000]\nprint('C9M9M0', cv_dual(['C9','M9','M0'],L))\nfor extra in [['C5'],['C7'],['C8'],['C10'],['C11'],['M11'],['C11','M11'],['C10','M10'],['C12','M12']]:\n    print('+',extra, cv_dual(['C9','M9','M0']+extra,L))\nprint('C11M11M0', cv_dual(['C11','M11','M0'],L))\nprint('C11M11M0C9', cv_dual(['C11','M11','M0','C9'],L))"
}
```


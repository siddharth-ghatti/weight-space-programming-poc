## reasoning summary

Linear ridge is still the strongest here at 97.8 to 97.9, since LDA and kernel ridge didn't beat the 97.5 mark. Next I'm weighing whether pooling should leave out the [CLS] and [SEP] tokens, which might help a little, though testing that would need token-level data I don't have yet.

I'm going to check whether the remaining feature options are worth more testing. The next idea is adding CLS-token states from intermediate layers to the joint ridge, but the earlier simulation showed adding more mean layers gave the same 97.86, so I think we're at a plateau. The open question is robustness on the test set, so I'll run repeated CV over several fold seeds and a few λ values to choose a stable setting, and also compare z-scored features against raw ones.



Linear ridge (97.8–97.9%) beats both shrinkage LDA and RBF kernel ridge (97.5% each), so I'm keeping the linear head. Next I'm running repeated cross-validation over different fold seeds to pick a robust λ and check whether standardizing features helps.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab); Y=np.eye(8)[lab]\nF=dict(np.load(WORK/'prep_feats.npz')); A=np.load(WORK/'v2_layer_scales.npy'); abar=A.mean(0)\nXs=np.concatenate([F[p][:,KEEP]*(A[:,k]/abar[k])[:,None] for k,p in enumerate(['m2','m6','m10'])]+[F['c12'][:,KEEP]],1)\ndef make_folds(seed):\n    rng=np.random.RandomState(seed); fo=np.zeros(N,int)\n    for c in range(8):\n        idx=np.where(lab==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%5\n    return fo\nlams=[50,100,150,250,400,700]\nres={('std',l):[] for l in lams}; res.update({('raw',l):[] for l in [1,3,10,30]})\nfor seed in range(1,6):\n    fo=make_folds(seed)\n    for mode in ['std','raw']:\n        Ks=[]\n        for f in range(5):\n            tr=fo!=f; te=~tr\n            mu=Xs[tr].mean(0); sd=(Xs[tr].std(0)+1e-6) if mode=='std' else np.ones(Xs.shape[1])\n            Z=(Xs[tr]-mu)/sd; Ze=(Xs[te]-mu)/sd; Ks.append((Z@Z.T,Ze@Z.T,tr,te))\n        for lam in (lams if mode=='std' else [1,3,10,30]):\n            o=np.zeros((N,8))\n            for K,Ke,tr,te in Ks:\n                al=np.linalg.solve(K+lam*np.eye(len(K)),Y[tr]-Y[tr].mean(0)); o[te]=Ke@al\n            res[(mode,lam)].append((o.argmax(1)==lab).mean())\nfor k,v in res.items(): print(k, np.round(np.mean(v),4), np.round(np.std(v),4))\n"
}
```


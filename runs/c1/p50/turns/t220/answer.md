## reasoning summary

The carried residual at the pool scales with per-sample k, so it's roughly linear with a noise term. Next I'm going to write a generic cross-validation helper that takes a custom linear-feature function for each fold, so I can test the A, C, and D variants.

For the C variant, I'd fit a joint model on each training fold with the old features, use its linear-block coefficients to build an 8-column score scaled by per-sample k, then refit with the new basis, with a large penalty constant so the penalty is negligible. D would do the same, but with the weights from the drop-46 features. Each fold then needs two joint fits, so roughly 20 seconds per seed, and I'd run 3 seeds of 10-fold.



The carried RES at the pool input has R² = 0.99, and its per-sample scale k_i varies by 15%, which I've saved. Next I'm running the simulations with that k_i noise added to the 8-dim carried scores.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json, time\nfrom lib7 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS); Y=np.eye(8)[y]\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\nfo=np.load('feats_seq1024_depth.npz'); Mo,Co=fo['M9'].astype(np.float64),fo['C9'].astype(np.float64)\nk=np.load('carry_k.npy')\ndef prep(M,C,tr,c=10,na=1008):\n    mu=M[tr].mean(0); Z=pool_z(M,C,c,mu); Zt=Z[tr]; Wf=anchor_dirs(Zt); Kc=gelu(Zt@Wf.T); Kc=.5*(Kc+Kc.T)\n    piv=pivchol(Kc,na); return feats(Z,Wf[piv]), Kc[np.ix_(piv,piv)]\ndef run(seed, kf=10, lam=.05, c0=.5):\n    fo_=folds(y,kf,seed); pa=np.zeros(1496,int); pc=pa.copy(); pd=pa.copy(); pd0=pa.copy()\n    for i in range(kf):\n        tr=fo_!=i; te=~tr; Lin=lin_feats(M0,tr)\n        Pn,Kn=prep(M,C,tr); Po,Ko=prep(Mo,Co,tr)\n        cn,bn=fit(Pn[tr],Lin[tr],Y[tr],Kn,lam,c0); co,bo=fit(Po[tr],Lin[tr],Y[tr],Ko,lam,c0)\n        pa[te]=(np.c_[Pn,Lin][te]@cn+bn).argmax(1)\n        for (W8,dest,kk) in [(co[1008:],pc,k),(cn[1008:],pd,k),(cn[1008:],pd0,np.ones(1496))]:\n            L8=(Lin@W8)*kk[:,None]\n            c8,b8=fit(Pn[tr],L8[tr],Y[tr],Kn,lam,100.0)\n            dest[te]=(np.c_[Pn,L8][te]@c8+b8).argmax(1)\n    return [(p==y).mean() for p in (pa,pc,pd,pd0)]\nt=time.time()\nR=[run(s) for s in range(10,14)]\nR=np.array(R); print(np.round(R,4)); print('A joint, C oldcarry+k, D newcarry+k, D0 newcarry nok', R.mean(0).round(4), round(time.time()-t))"
}
```


## reasoning summary

I'm putting together the edited features: the final CLS hidden state at layer 12 for sample 1496, the edited mean hidden state at layer 11, and the layer 11 CLS state. Next I want to check how well the late layers predict on their own, using ridge cross-validation on the final CLS coordinates with R excluded, and compare that against the base m11 score of 95.9%. I'm also starting to think about fusing the out-of-fold carried logits from the base model with the ridge outputs.

I'm weighing how to fuse the late-layer ridge with the carried base logits, since the R coordinates at the final CLS are roughly a scaled, shifted copy of the sentence scores. To choose the mixing weight, I'll simulate the fusion with out-of-fold scores and per-sentence scales, and keep R out of the late ridge features to avoid leakage.



I'm assembling the edited model's [CLS] features for all 1496 sentences, then evaluating ridge CV on the non-R coordinates alone and fusing with the out-of-fold logits to choose α.



## tool_use: run_python

```json
{
 "code": "lab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nR = [601, 222, 88, 83, 444, 506, 254, 495]; keep=np.setdiff1d(np.arange(768),R)\ndef cat(kind,L):\n    return np.concatenate([np.load(WORK/f'pass_3/{kind}_hidden_L{L}.npy'),np.load(WORK/f'pass_4/{kind}_hidden_L{L}.npy'),np.load(WORK/f'pass_5/{kind}_hidden_L{L}.npy')])\nE12=cat('cls',12); E11m=cat('mean',11); E11c=cat('cls',11)\nnp.save(WORK/'ed1_cls12.npy',E12); np.save(WORK/'ed1_mean11.npy',E11m); np.save(WORK/'ed1_cls11.npy',E11c)\nM=np.load(WORK/'base_mean.npy'); D=np.load(WORK/'D.npy'); b0=np.load(WORK/'b0.npy')\ns_in=M[2]@D.T+b0\n# check R decode on all\nr=E12[:,R]; rc=r-r.mean(1,keepdims=True); sc=s_in-s_in.mean(1,keepdims=True)\na=(rc*sc).sum(1)/(sc**2).sum(1); print('scale a', a.mean().round(3), a.std().round(3), 'agree', (r.argmax(1)==s_in.argmax(1)).mean())\nnp.save(WORK/'ed1_a.npy',a)\ndef fit_ridge(Xt,Yt,lam):\n    mu=Xt.mean(0); sd=Xt.std(0)+1e-6; Z=(Xt-mu)/sd\n    W=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Yt-Yt.mean(0)))\n    return lambda X: ((X-mu)/sd)@W\ndef oof(X,lam):\n    o=np.zeros((N,8))\n    for f in range(5):\n        tr=folds!=f; te=~tr; o[te]=fit_ridge(X[tr],Y[tr],lam)(X[te])\n    return o\nacc=lambda S:round((S.argmax(1)==lab).mean(),4)\ns_oof=oof(M[2][:,keep],300)\nprint('carried oof acc', acc(s_oof))\nfor lam in [30,100,300]:\n    late=oof(E12[:,keep],lam)\n    print('late E12 nonR lam',lam, acc(late), 'fusion', [ (al,acc(al*s_oof*(a/a.mean())[:,None]+late)) for al in [0.5,0.75,1,1.25,1.5,2]])\nfor lam in [100,300]:\n    print('base m11 lam',lam, acc(oof(M[11][:,keep],lam)), ' edited m11', acc(oof(E11m[:,keep],lam)), 'edited 2*m11+c11', acc(oof(2*E11m[:,keep]+E11c[:,keep],lam)))\n"
}
```


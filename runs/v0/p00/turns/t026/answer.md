## reasoning summary

The position 1 error is about 5%, mostly in magnitude, and that's small enough to leave alone. I could weight position 1 more heavily in the fit, but it's only one of roughly 31 tokens, so it isn't worth changing. I'm now considering whether any improvements are worth the remaining budget, focusing on exp

I'm checking whether the classifier ties could matter when a sentence scores all zeros, and concluding that case isn't realistic. I'm also confirming the final layer norm: the CLS residual is about 27.6 in norm from non-DF content, while the DF contribution is roughly ±1.5, so after standardizing the DF signal stays around unit scale.

I'm checking the intermediate activations in layers 2 through 11: their base weights are still in place, but their outputs get zeroed by the output projection, so the question is whether anything could overflow to infinity and turn the zero product into NaN. With standardized inputs the values should stay in a normal range, so the risk looks negligible.

I'm settling on leaving the attention and intermediate weights as they are, since the base weights already give finite values and there's no real NaN risk to address. Zeroing them would be harmless but unnecessary, so I'm skipping that edit.

I'm settling on the gain of 4 and the 0.3 scaling, which puts the final DF values around ±1.2, so the signal stays well-scaled. The blend3 variant scores 97.42 against 97.30 for the current blend, which is within noise, so I'll keep the current version. I'm also checking whether the blend's confusion matrix shows any class being systematically over-predicted that would need a bias correction, though the NB matrix looked balanced earlier.

I'm checking how the CLS and SEP columns were handled: they were left in the CV scoring as constant per-class offsets, but I zeroed them in the network, so the two versions aren't quite the same.

I'm checking whether the CV accuracy still holds with the CLS and SEP columns zeroed, so the scoring matches what the network actually does.



The detectors hold up: special tokens stay near zero out to position 511, and seen tokens keep their direction even at position 127. Since CV kept nonzero scores for [CLS]/[SEP] while the network zeroes them, I'm re-running CV with them zeroed so the estimate matches the build.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522; N=len(P)\nn=np.array([len(t) for t in ids]); Xtf=sp.diags(1.0/n)@X\ndef nb(tr,alpha=0.3):\n    cnt=np.zeros((V,8))\n    for c in range(8): cnt[:,c]=np.asarray(X[tr][y[tr]==c].sum(0)).ravel()\n    lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*V); return lp-lp.mean(1,keepdims=True)\ndef ridge(tr,Xf,lam):\n    Xc=Xf[tr]; Yc=np.eye(8)[y[tr]]\n    Kk=(Xc@Xc.T).toarray(); A=np.linalg.solve(Kk+lam*np.eye(len(tr)), Yc-Yc.mean(0))\n    S=np.asarray(Xc.T@A); return S-S.mean(1,keepdims=True)\nres={}\nallpred=[]\nfor seed in [0,1,2]:\n    rng=np.random.RandomState(seed); perm=rng.permutation(N); folds=np.array_split(perm,5)\n    pred=np.zeros(N,int); pred0=np.zeros(N,int)\n    for f in folds:\n        tr=np.setdiff1d(np.arange(N),f)\n        S=nb(tr)/4+ridge(tr,Xtf,3e-3)\n        pred[f]=(X[f]@S).argmax(1)\n        S0=S.copy(); S0[[0,101,102]]=0\n        pred0[f]=(X[f]@S0).argmax(1)\n    res.setdefault('with cls/sep',[]).append((pred==y).mean()); res.setdefault('zeroed',[]).append((pred0==y).mean())\n    allpred.append(pred0)\nfor k,v in res.items(): print(k, np.round(v,4), np.mean(v))\ncm=np.zeros((8,8),int)\nfor a_,b_ in zip(np.tile(y,3),np.concatenate(allpred)): cm[a_,b_]+=1\nprint(cm)\nf1=[2*cm[c,c]/(cm[c].sum()+cm[:,c].sum()) for c in range(8)]; print('per-class F1', np.round(f1,3))"
}
```


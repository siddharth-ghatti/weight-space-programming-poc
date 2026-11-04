## reasoning summary

Bigrams would add only about 0.3 to 0.7 points, and with M=40 it's roughly 0.3%, which doesn't justify the risk of building a previous-token head blind, so I'm dropping them. For the unigram model, the blend choice looks fine as it is. The bigger candidate is weighted pooling, using attention with token-specific weights in place of the uniform mean, with CLS attention as one possible source of those weights.

I've ruled out weighted pooling as a gain: any per-token weighting just rescales each token's score vector, so it reduces to choosing the table directly. That leaves improving the table itself, through ridge strength, NB smoothing, or blend weights, and I'm going to try a few closed-form classifiers next, starting with ridge on IDF-weighted term features.

Next I'm checking ridge on raw counts against TF features, since the argmax over summed token scores doesn't change with normalization, so either fit can ship as the same table. I'll also look at LDA-style regularized least squares with class-balanced targets and average a few tables from different settings. Given the roughly 0.4% standard error on 1496 examples, I'll keep the comparison restrained across seeds and folds.

Next I'm checking ridge on counts against ridge on L2-normalized counts across a few regularization strengths, plus blends with NB at different weights.



Bigrams add only 0.3–0.7% in CV, which isn't worth a blind previous-token head, so I'm sticking with unigrams and comparing a few closed-form variants over 3 seeds × 5 folds, keeping the roughly ±0.4% CV noise in mind.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522; N=len(P)\nn=np.array([len(t) for t in ids]); Xtf=sp.diags(1.0/n)@X\nnrm=np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); Xl2=sp.diags(1.0/nrm)@X\ndef nb(tr,alpha=0.3):\n    cnt=np.zeros((V,8))\n    for c in range(8): cnt[:,c]=np.asarray(X[tr][y[tr]==c].sum(0)).ravel()\n    lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*V); return lp-lp.mean(1,keepdims=True)\ndef ridge(tr,Xf,lam):\n    Xc=Xf[tr]; Yc=np.eye(8)[y[tr]]\n    Kk=(Xc@Xc.T).toarray(); A=np.linalg.solve(Kk+lam*np.eye(len(tr)), Yc-Yc.mean(0))\n    S=np.asarray(Xc.T@A); return S-S.mean(1,keepdims=True)\nres={}\nfor seed in [0,1,2]:\n    rng=np.random.RandomState(seed); perm=rng.permutation(N); folds=np.array_split(perm,5)\n    for f in folds:\n        tr=np.setdiff1d(np.arange(N),f)\n        T={'nb':nb(tr)}\n        for lam in [1e-3,3e-3,1e-2]: T[f'rtf{lam}']=ridge(tr,Xtf,lam)\n        for lam in [0.3,1,3]: T[f'rcnt{lam}']=ridge(tr,X,lam)\n        for lam in [0.03,0.1,0.3]: T[f'rl2{lam}']=ridge(tr,Xl2,lam)\n        T['blend']=T['nb']/4+T['rtf0.003']\n        T['blend2']=T['nb']/8+T['rtf0.003']\n        T['blend3']=T['nb']/2+T['rtf0.003']\n        T['blend_l2']=T['nb']/4+T['rtf0.003']+ T['rl20.1']*np.linalg.norm(T['rtf0.003'])/np.linalg.norm(T['rl20.1'])\n        for k,S in T.items():\n            res.setdefault(k,[]).append(((X[f]@S).argmax(1)==y[f]).mean())\nfor k,v in res.items(): print(f'{k:12s} {np.mean(v):.4f}')"
}
```


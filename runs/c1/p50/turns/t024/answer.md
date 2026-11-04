## reasoning summary

I'm reconsidering the random feature scale: at scale 1 the tanh is already saturating, so something like 0.1 to 0.3 would keep it closer to linear with a small cubic term. I also need the linear component included, since the classifier only sees the pooled tanh outputs and a near-linear tanh would otherwise lose that signal.

I'm weighing whether the anchor-based approach can stand in for the RBF kernel. Anchoring the pooler units to training points, with tanh shaped by a bias, might approximate the exponential kernel shape. I want to test that offline with ridge on the resulting features.

I'm thinking about the pooler as the nonlinear layer, since it's just tanh of a linear map followed by a linear classifier, and the classifier is only 8×768. To get close to kernel ridge performance, the 768 features need to span the kernel's leading eigenfunctions, and Nyström with 768 anchors should do that reasonably well. I'm also noting that tanh of dot products is a sigmoid kernel, which isn't guaranteed positive definite but often works in practice, so I want to test it offline with anchors drawn from training data.

I'm going to check whether the anchors need to come only from each training fold, since test points used as anchors would look closer to them than new data would. I'll set that up per fold to keep the estimate honest. I also want to try random tanh features on whitened PCA inputs at several scales, plus the anchor-based version.

I'm weighing whether the pooling layer's FFN could add a useful nonlinearity, but it would just feed into the same 768-dim projection, so it only helps if it adds expressivity. I'm going to test the nonlinear head variants offline to see how much they actually gain.



Mean features compress well: PCA to 256–384 dims keeps nearly all the RBF accuracy, which frees attention heads for other pooling patterns. The tanh random-feature scales I tried were too large and saturated, so next I'll try a proper tanh head with smaller scales and anchor-based units.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef whiten_fit(X, r=384, eps=1e-3):\n    mu=X.mean(0); U,S,Vt=np.linalg.svd(X-mu,full_matrices=False); \n    P=Vt[:r].T/ (S[:r]/np.sqrt(len(X)))  # unit-variance PCA\n    return mu,P\ndef head_cv(X, y, kind, k=10, seeds=(0,), lams=(0.01,0.1,1), **kw):\n    res={lam:[] for lam in lams}\n    for s in seeds:\n        f=folds(y,k,s); S={lam:np.zeros((len(y),8)) for lam in lams}\n        for i in range(k):\n            tr=f!=i; te=f==i\n            mu,P=whiten_fit(X[tr],kw.get('r',384)); Z=(X-mu)@P; Z/=np.sqrt(Z.shape[1])  # rms norm ~1\n            if kind=='anchor':\n                rng=np.random.RandomState(i); A=Z[tr][rng.choice(tr.sum(),768,replace=False)]\n                G=Z@A.T; F=np.tanh(kw['a']*G+kw['b'])\n            elif kind=='rf':\n                rng=np.random.RandomState(i); Wr=rng.randn(Z.shape[1],768)*kw['s']; F=np.tanh(Z@Wr*np.sqrt(Z.shape[1])/np.sqrt(Z.shape[1]))\n            elif kind=='lin': F=Z\n            for lam in lams:\n                W,b=ridge_fit(F[tr],y[tr],lam); S[lam][te]=F[te]@W+b\n        for lam in lams: res[lam].append((S[lam].argmax(1)==y).mean())\n    return {lam:round(float(np.mean(v)),4) for lam,v in res.items()}\nX0=M(0)\nprint('lin', head_cv(X0,y,'lin',lams=(0.1,1,3,10)))\nfor s in [0.5,1,2]:\n    print('rf s',s, head_cv(X0,y,'rf',s=s,lams=(0.01,0.1,1)))\nfor a,b in [(1,0),(2,-1),(3,-2),(5,-4)]:\n    print('anchor',a,b, head_cv(X0,y,'anchor',a=a,b=b,lams=(0.01,0.1,1)))"
}
```


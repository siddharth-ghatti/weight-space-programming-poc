## reasoning summary

The kernel works out to roughly exp(G), with squared norms between 0.56 and 2.28, and the cosine-exp version reaches 97.53%. The tanh kernel is unstable as full kernel ridge regression at small λ, but at larger λ it holds up, reaching 97.3%. The 768 Nyström anchors may be the bottleneck, so I'll test that next.

I'm going to check whether the 768-anchor Nyström version with an exp kernel reaches about 97.4%, since that would show the exp shape matters. If it does, I could approximate exp with a tanh tail. I'll also test the tanh tail with ridge, without standardization, at a properly chosen λ.



The exp kernel reaches 97.5% as exact kernel ridge, and tanh(G−1)+1 reaches 97.3% with enough regularization. Next I'll test whether the 768 anchors are the bottleneck, by running the exp kernel and the tanh variant with properly scaled regularization on 768 anchors.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef nys_cv(X, y, fn, k=10, seeds=(0,), lams=(0.01,0.1,1), nanc=768, cosn=True, scale_lam=True):\n    res={lam:[] for lam in lams}\n    for s in seeds:\n        f=folds(y,k,s); S={lam:np.zeros((len(y),8)) for lam in lams}\n        for i in range(k):\n            tr=f!=i; te=f==i\n            mu=X[tr].mean(0); Z=X-mu\n            Z=Z/np.linalg.norm(Z,axis=1,keepdims=True) if cosn else Z/np.sqrt((Z[tr]**2).sum(1).mean())\n            rng=np.random.RandomState(i+100*s); A=Z[tr][rng.choice(tr.sum(),nanc,replace=False)]\n            F=fn(Z@A.T)\n            v=F[tr].var(0).sum()\n            for lam in lams:\n                W,bb=ridge_fit(F[tr],y[tr],lam*v if scale_lam else lam); S[lam][te]=F[te]@W+bb\n        for lam in lams: res[lam].append((S[lam].argmax(1)==y).mean())\n    return {lam:round(float(np.mean(v)),4) for lam,v in res.items()}\nX=M(0)\nL=(1e-4,1e-3,1e-2,1e-1)\nprint('exp nys768', nys_cv(X,y,lambda G: np.exp(G),lams=L))\nprint('exp nys768 x2', nys_cv(X,y,lambda G: np.exp(2*G),lams=L))\nprint('exp nys1196', nys_cv(X,y,lambda G: np.exp(G),lams=L,nanc=1196))\nfor a,b in [(0.5,-3),(1,-3),(1,-4),(2,-4)]:\n    print('tanh',a,b, nys_cv(X,y,lambda G: np.tanh(a*G+b),lams=L))"
}
```


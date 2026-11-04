## reasoning summary

The "unif" block, which is the M0 projection onto the class directions plus PCA components, improves the kernel to about 98% when weighted at 0.5. That gain looks like just re-weighting the discriminative subspace, since those directions were fit on the labels. That would make the cross-validation optimistic, because the directions were fit on all the data, so I need to check whether the leakage is inflating these numbers.

The fold-by-fold check needs redoing: the direction matrix has to be refit on each training fold, not once on all the data. I'll compute the unif projection per fold to see whether the 98% gain survives. If it does, the same linear map could be learned directly through the attention value and output weights, so I'll test that next.

Any linear map A of the pooled vector is reachable through the value and output projections, so the question is which A helps the kernel most. One option is to boost the class-direction subspace, another is a whitening-style or LDA-like transform. I'll test this with nested CV, fitting the ridge directions on each training fold and weighting them at several levels.



The pooling layer's position pools hurt, while the uniform block lifts CV to about 98%, so I'm re-running it with nested CV to avoid optimistic bias from class directions fit on all labels.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\ngelu=lambda t: 0.5*t*(1+erf(t/np.sqrt(2)))\nd=load_base(); y=labels(); M0=d['mean_hidden_L0']\ndef cosnorm(X, mu=None):\n    Z=X-(X.mean(0) if mu is None else mu); Z=Z-Z.mean(1,keepdims=True); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef kern(Z1,Z2,a=1.0): S=Z1@Z2.T; return S+(gelu(a*S)+gelu(-a*S))/(a*a)\ndef nb(x): x=x-x.mean(0); return x/np.sqrt((x**2).sum(1).mean())\ndef nested_cv(X, y, w, k=10, seeds=(0,1,2), lam_r=3.0, lam_k=0.03, ndir='ridge', orth=False):\n    accs=[]; X=nb(X)\n    for s in seeds:\n        f=folds(y,k,s); P=np.zeros(len(y),int)\n        for i in range(k):\n            tr=f!=i; te=f==i\n            Wr,_=ridge_fit(X[tr],y[tr],lam_r)\n            if orth: Qd,_=np.linalg.qr(Wr); proj=X@Qd\n            else: proj=X@Wr\n            sc=np.sqrt(((proj[tr]-proj[tr].mean(0))**2).sum(1).mean())\n            F=np.concatenate([X, w*(proj-proj[tr].mean(0))/sc],1)\n            Z=cosnorm(F, F[tr].mean(0))\n            K=kern(Z[tr],Z[tr]); Y=np.eye(8)[y[tr]]\n            A=np.linalg.solve(K+lam_k*np.eye(tr.sum()),Y-Y.mean(0)); P[te]=(kern(Z[te],Z[tr])@A).argmax(1)\n        accs.append((P==y).mean())\n    return round(float(np.mean(accs)),4)\nfor w in [0,0.3,0.5,0.7,1.0]:\n    print('w',w,'ridge-scores 10f',nested_cv(M0,y,w),'20f',nested_cv(M0,y,w,k=20,seeds=(0,1)))\nfor w in [0.5,1.0]:\n    print('w',w,'orth-dirs 10f',nested_cv(M0,y,w,orth=True))"
}
```


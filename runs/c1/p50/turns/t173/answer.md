## reasoning summary

I'm weighing the anchor-count numbers: going to m=768 would cost about 0.2% from the pool alone, before any content pruning, which would put CV near 97.9–98.0 and seems too risky. I'm also looking at the pool layer's kept units, where a handful of units with weight norm 1.0 stand out and I'm wondering whether that's a linear pass-through effect.

I'm now asking whether the pool layer can do more with the same units. Its attention is uniform mean pooling, and since attention is linear in the values, it can't add nonlinearity; the FFN is the only nonlinear part. I'm also considering better anchors than training points, such as k-means centroids or eigen-directions, given that random anchors lose 0.4% and pivoted Cholesky did best of those tried.

I'm considering whether the kernel units could be spread across content layers instead of living only in the pool layer, perhaps gated on a CLS-specific feature so they stay silent on other tokens. That seems possible in principle but too complex at this stage. The realistic choices are submitting the current 63.8M model at CV 98.37, or trimming m to 896 for about a 3% saving, which costs a little accuracy. Since the pool layer needs roughly 1000 units, m is effectively bounded by that requirement.

I'm thinking about whether the kernel anchors could be chosen better than training points, such as with k-means centroids, since random anchors lose accuracy and pivoted Cholesky does best so far. I'm also wondering if the units could be placed more cleverly, though the pool layer still seems to need around 1000 of them.

I'm wondering whether the linear part of the classifier, the direct map from the mean features, could be carried by the pool's attention values rather than by FFN units, so the FFN only has to model the nonlinear residual. In the current fit only the GELU features are used, with the linear term coming from M0, so adding a linear M9 term would mean routing it through the attention output into the residual dimensions, though that output passes through LN before the FFN.

I'm checking whether the linear M9 term could go through the pool's attention output and the residual stream, since that path is free. I want to test whether it helps with fewer anchors, around 752, and whether a different regularization strength is needed. The gain ceiling looks like about 6% of parameters, while the margin over the 97.8 threshold is only about 0.57%, so the risk weighs heavily.

I'm going to check whether the pool layer, with 752 anchors and a linear M9 term fit jointly, can reach about 98.3 on the current features. If it does, I'll need to test how much accuracy the content pruning costs at m=768.



Anchor count drives the score: 1008 anchors give 98.37, down to 496 giving 97.62, with a pool cost of about 0.2% at m=768. Next I'll try recovering that with a linear M9 term, retuned lam, and k-means anchors.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nF={k:v.astype(np.float64) for k,v in np.load('feats_seq1024_depth.npz').items()}\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\ndef cv3(na, extra=('M0',), lam=0.02, c0=0.5, seeds=(0,1,2), mode='piv'):\n    M=F['M9']; C=F['C9']; res=[]\n    for sd in seeds:\n        f=folds(y,10,sd); Y=np.eye(8)[y]; pred=np.zeros(len(y),int)\n        for i in range(10):\n            tr=f!=i; te=~tr; mu=M[tr].mean(0)\n            Z=pool_z(M,C,10.0,mu); Zt=Z[tr]; Wf=anchor_dirs(Zt)\n            if mode=='piv':\n                Kc=gelu(Zt@Wf.T); Kc=.5*(Kc+Kc.T); piv=pivchol(Kc,na); Wd=Wf[piv]\n            else:\n                # per-class kmeans (few Lloyd steps on cosine) -- closed-form-ish clustering of activations\n                rng=np.random.RandomState(sd); A=Zt/np.linalg.norm(Zt,axis=1,keepdims=True); yt=y[tr]; cents=[]\n                for cl in range(8):\n                    Ac=A[yt==cl]; kc=na//8; Cn=Ac[rng.choice(len(Ac),kc,replace=False)]\n                    for it in range(10):\n                        asg=(Ac@Cn.T).argmax(1)\n                        for j in range(kc):\n                            if (asg==j).any(): v=Ac[asg==j].mean(0); Cn[j]=v/np.linalg.norm(v)\n                    cents.append(Cn)\n                Cn=np.concatenate(cents); Cn[:,RES]=0; Wd=Cn/np.linalg.norm(Cn,axis=1,keepdims=True)/np.sqrt(768)\n            Kaa=gelu(Wd@Wd.T*768); Phi=feats(Z,Wd)\n            Lin=np.concatenate([lin_feats(F[e],tr) for e in extra],1)\n            coef,b=fit(Phi[tr],Lin[tr],Y[tr],Kaa,lam,c0/len(extra))\n            pred[te]=(np.concatenate([Phi[te],Lin[te]],1)@coef+b).argmax(1)\n        res.append((pred==y).mean())\n    return round(np.mean(res),4), np.round(res,4)\nprint('752 piv base', cv3(752))\nprint('752 piv +M9', cv3(752, extra=('M0','M9')))\nprint('752 piv lam.01', cv3(752, lam=0.01))\nprint('752 piv lam.005', cv3(752, lam=0.005))\nprint('752 kmeans', cv3(752, mode='km'))\nprint('1008 kmeans', cv3(1008, mode='km'))\n"
}
```


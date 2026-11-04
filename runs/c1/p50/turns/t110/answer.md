## reasoning summary

The cosine values sit near zero, with the 99th percentile around 0.43, so applying GELU(3s−1) would push almost every feature to roughly the same negative value. I'd rather use gelu(s) with a=1, b=0, which already gave 98.03 on this fold.

I'm weighing whether the anchor-based primal approximation can match the exact dual result: with a=1, b=0, c0=0 it reaches 97.86 against roughly 98.17 for exact KRR, and the 0.3% gap looks like it comes from using 1000 anchors instead of the full training set. The surprising part is that adding the linear block hurts in the primal (97.86 down to 97.06) while it helps in the dual (98.53), so I suspect a weighting mismatch between the two blocks' diagonals.

The linear block should be a simple addition, so the drop from 97.86 to 97.06 in the primal is odd. My guess is the normalization: the anchor features Φ may already act like a squared kernel, so mixing in the linear term at the same weight doesn't mean what it does in the dual.

The primal ridge here doesn't have the same penalty as KRR, since ΦΦᵀ behaves like a smoothed K² and the linear block gets lost in it. To get the exact KRR answer, I'd need a penalty of λαᵀK_aa α on the anchor coefficients, which gives α = (ΦᵀΦ + λK_aa)⁻¹ΦᵀY, the Nyström form. The linear part would need its own penalty scaled by 1/c0.

I'm going to write the Nyström-regularized ridge with a block penalty of diag(K_aa, I/c0), since the anchor features and the linear readout weights each need their own penalty. I'm also checking whether the GELU kernel on anchors is PSD enough for that penalty to be valid, or whether the exp kernel should be used there instead.

I'm weighing whether a GELU-based kernel is positive semi-definite enough to serve as the penalty on the anchors. One option is to pair each anchor with GELU(-s) as a second unit, which makes the kernel symmetric, but that halves how many anchors I can afford. A simpler route might be scaling the input as GELU(a·s + b), with the kernel handled through jitter.

I'm going to set up the Nyström-penalized fit with a GELU kernel and check whether the linear block matters, first using 1000 anchors and then all 1346 training points.



The off-diagonal cosines are mostly small, so the GELU kernel is nearly constant and degenerate, and GELU(s) performs as well as exp. The leftover gap comes from the plain ridge penalty, so I'm switching to a Nyström/RKHS penalty with a separate one for the linear M0 block.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nZ9=cn(d['mean_hidden_L9']); C0=cen(d['mean_hidden_L0'])\ndef nys_cv(a=1.0,b=0.0,nanch=1000,lam=0.005,c0=0.5,seeds=(0,1,2),k=10,Z=Z9,C=C0,yv=y,jit=1e-6):\n    accs=[]; n=len(yv); Yv=np.eye(8)[yv]\n    for s in seeds:\n        f=folds(yv,k,s); S=np.zeros((n,8)); rng=np.random.RandomState(100+s)\n        for i in range(k):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            anc=rng.choice(tr, min(nanch,len(tr)), replace=False)\n            Phi=gelu(a*(Z@Z[anc].T)+b); Kaa=gelu(a*(Z[anc]@Z[anc].T)+b)\n            na=len(anc)\n            if c0>0:\n                F=np.hstack([Phi, C]); Pen=np.zeros((na+C.shape[1],)*2); Pen[:na,:na]=Kaa+jit*np.eye(na); Pen[na:,na:]=np.eye(C.shape[1])/c0\n            else:\n                F=Phi; Pen=Kaa+jit*np.eye(na)\n            mu=F[tr].mean(0); Fc=F-mu\n            Wt=np.linalg.solve(Fc[tr].T@Fc[tr]+lam*Pen, Fc[tr].T@(Yv[tr]-Yv[tr].mean(0)))\n            S[te]=Fc[te]@Wt\n        accs.append((S.argmax(1)==yv).mean())\n    return round(float(np.mean(accs)),4)\nprint('exact dual K9:',krr_cv(np.exp(Z9@Z9.T-1),y,[0.005],k=10,seeds=(0,1,2)),' K9+.5lin0:',krr_cv(np.exp(Z9@Z9.T-1)+0.5*C0@C0.T,y,[0.005],k=10,seeds=(0,1,2)))\nprint('exact dual gelu:',krr_cv(gelu(Z9@Z9.T),y,[0.003,0.01],k=10,seeds=(0,1,2)),' +.5lin0:',krr_cv(gelu(Z9@Z9.T)+0.5*C0@C0.T,y,[0.003,0.01],k=10,seeds=(0,1,2)))\nfor nanch in [1000,1400]:\n    for c0 in [0.0,0.5]:\n        print('nys',nanch,'c0',c0,[nys_cv(nanch=nanch,c0=c0,lam=l) for l in [0.003,0.01,0.03]])"
}
```


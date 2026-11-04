## reasoning summary

I'm checking the ideal Zs pseudo-cos, which lands at 97.97 against a target of 98.06, so normalization doesn't seem to be the real cause of the gap. The drop to 97.75 in run 48 must come from how the representation is built from x, and I'm comparing the restricted PC subsets for the M and C parts against the full setup.

I'm running the ablations now, starting with the restricted M and C parts against the full setup.



Pseudo-cosine nearly matches true cosine in the ideal geometry (97.97% vs 98.06%), so normalization isn't the source of the 0.3% loss. I'm ablating from the ideal setup to my construction to locate where it goes.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\ndef cv(Lin,R,lams=(10,30),b=10,t=4.5,wl=0.3):\n    rbar=np.linalg.norm(R,axis=1).mean(); G=(R@R.T)/np.linalg.norm(R,axis=1)[None,:]/rbar\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            Z=np.concatenate([wl*Lin,gelu(b*G[:,tr]-t)],1)\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nk2=384\nec,Vc=np.linalg.eigh(np.cov(C9.T)); Q=Vc[:,:k2]\nmC=C9.mean(0); sC=C9.std(0); Cs=(C9-mC)/sC\nMs=(M9-M9.mean(0))/M9.std(0)\n_,_,VtM=np.linalg.svd(Ms,full_matrices=False); sc=Ms@VtM[:k2].T\nprint('A Zs', cv(np.c_[Cs,Ms],np.c_[Cs,Ms]))\nprint('B Cs+sc384', cv(np.c_[Cs,sc],np.c_[Cs,sc]))\nCr=(C9-mC)@(np.eye(768)-Q@Q.T)/sC\nprint('C Cr+sc384', cv(np.c_[Cr,sc],np.c_[Cr,sc]))\nlam=0.3\nu=C9+lam*sc@Q.T; mu=u.mean(1,keepdims=True); sig=np.sqrt(((u-mu)**2).mean(1,keepdims=True))\nD=np.c_[Cr,sc]*sig.mean()/sig\nprint('D scaled by 1/sigma', cv(D,D))"
}
```


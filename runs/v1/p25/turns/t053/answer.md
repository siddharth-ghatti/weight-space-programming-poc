## reasoning summary

I'm weighing whether to keep the raw, unwhitened PC scores of the standardized M9 features, since the top components carry most of the variance and the ridge penalty works on that raw scale anyway. The linear part seems to need many M9 components to get close to the 97.64 target, so I'm considering packing 384 of them into the bottom 384 C9 directions, possibly even more.

I'm now setting up a full simulation where the unwhitened PC scores of the standardized M9 features get folded into the bottom C9 directions, with a kernel built on the top stdz-PC directions of C plus the M projection, and I want to run a CV on it to see how the C and M weighting behaves.

I'm checking how to project the whitened directions off the Q subspace, so the cross-talk term drops out. Rather than tune this by hand, I'll simulate the exact feature map an FFN would compute, using a linear projection followed by a GELU kernel with normalization folded into the weights.

I'm working out how the kernel units respond to the normalized projection, folding in the norm of each direction so the function only needs the fixed weights. Next I'm picking λ so the M part is large enough to dominate the tiny bottom-subspace variance without swamping the normalization of the whole vector.

I'm checking the M-part scale: with λ=0.1 the M norm is about 2.8 against roughly 0.7 of noise from the bottom C9 directions, so that mix is noticeably polluted. Since both sources carry useful information for the linear fit, I'll test λ at 0.1, 0.3, and 0.5 and compare how much each one loses.



The linear part needs about 384 M9 PCs (97.55% variance), so I'm encoding those scores into C9's bottom-384 directions with uniform attention, then applying LN. Next I'll simulate what the layer can compute and pick λ, β and θ.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nk2=384; k1=384\nec,Vc=np.linalg.eigh(np.cov(C9.T)); Q=Vc[:,:k2]\nmM=M9.mean(0); sM=M9.std(0); Ms=(M9-mM)/sM\n_,_,VtM=np.linalg.svd(Ms,full_matrices=False); PM=VtM[:k2].T      # 768xk2\nsc=Ms@PM\nmC=C9.mean(0); sC=C9.std(0); Cs=(C9-mC)/sC\n_,_,VtC=np.linalg.svd(Cs,full_matrices=False); PC_=VtC[:k1].T\ndef build(lam):\n    u=C9+lam*sc@Q.T\n    x=u-u.mean(1,keepdims=True); x/=np.sqrt((x**2).mean(1,keepdims=True)+1e-12); return x\ndef feats(x,lam):\n    Wc=(np.eye(768)-Q@Q.T)@(PC_/sC[:,None])   # 768xk1\n    Rc=x@Wc; Rm=x@Q/lam\n    return Rc,Rm\ndef cv(x,lam,lams,b=10,t=4.5,wl=0.3,wm=1.0,klin=True):\n    Rc,Rm=feats(x,lam); R=np.concatenate([Rc,wm*Rm],1)\n    rbar=np.linalg.norm(R,axis=1).mean()\n    xs=(x-x.mean(0))/(x.std(0)+1e-9)\n    res={l:0 for l in lams}; resl={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            Rn_tr=R[tr]/np.linalg.norm(R[tr],axis=1,keepdims=True)\n            Kf=gelu(b*(R@Rn_tr.T)/rbar-t)\n            Z=np.concatenate([wl*xs,Kf],1)\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\ndef cvlin(x,lams):\n    xs=(x-x.mean(0))/(x.std(0)+1e-9); res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            for l in lams:\n                Wt,bb=ridge_W(xs[tr],Y,l); res[l]+=((xs[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nfor lam in [0.05,0.1,0.3]:\n    x=build(lam)\n    print(lam,'lin',cvlin(x,[100,300,1000]),'kern',cv(x,lam,[10,30]))"
}
```


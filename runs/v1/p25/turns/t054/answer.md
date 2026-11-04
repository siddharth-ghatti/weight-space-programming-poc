## reasoning summary

I'm weighing the linear model, which standardizes each dimension of x, though x's dimensions are mixtures, so that isn't the same as standardizing C9 and M9. The linear fit does well at λ=300, but the kernel version is worse, likely because its linear part uses a weaker λ setting, so the kernel combinations will need retuning.

I'm working out how to define the linear features for the ridge so they match the standardized concatenation, since per-dimension standardization of x doesn't give the same scale as standardizing C9 and M9. The previous kernel setup used a weaker linear weight, so I'll need to retune that weight and the ridge penalty together.

I'm sketching the linear features as a 768-dimensional set, scaled per dimension by C9's standard deviation, and pairing them with the M9 PC scores divided by λ. I'll define that, then tune the weights.

I'm estimating the scale of the features: the norms of the C9 and M9 parts come out to roughly 15 and 8, which puts σ_u near 0.61 and the scaling factor around 1.6. Next I'm rebuilding the linear features so each part is scaled by a shared σ̄, with the C9 directions and the M9 PC scores weighted separately.

I'm settling on the linear features as a rotation of x: the C9 directions outside the Q subspace, scaled by the C9 standard deviations, paired with the M9 PC scores divided by λ, all multiplied by a shared σ̄ so the geometry matches the standardized concatenation. Then the kernel uses those as its linear part, and I'll sweep the ridge λ and the linear weight w_l together.



The linear part got 97.68% at λ=0.3 while the kernel lagged, so I'm rebuilding the features to mirror standardized [C9, M9-PCs] using fixed linear maps, then retuning.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nk2=384; k1=384\nec,Vc=np.linalg.eigh(np.cov(C9.T)); Q=Vc[:,:k2]\nmM=M9.mean(0); sM=M9.std(0); Ms=(M9-mM)/sM\n_,_,VtM=np.linalg.svd(Ms,full_matrices=False); PM=VtM[:k2].T\nsc=Ms@PM\nmC=C9.mean(0); sC=C9.std(0)\nlam=0.3\nu=C9+lam*sc@Q.T; mu_u=u.mean(1,keepdims=True); sig=np.sqrt(((u-mu_u)**2).mean(1,keepdims=True))\nx=(u-mu_u)/sig\nprint('sigma_u pct',np.percentile(sig,[1,50,99]))\nsb=sig.mean()\nPn=np.eye(768)-Q@Q.T\nLC=(x@Pn)*sb/sC            # ~ C9 (nonQ part) / sC\nLM=(x@Q)*sb/lam            # ~ sc\nLCc=LC-LC.mean(0); \n_,_,VtC=np.linalg.svd(LCc,full_matrices=False); RC=LCc@VtC[:k1].T\nnp.savez('sim_x.npz',LC=LC,LM=LM,RC=RC)\ndef cv(lams,wl,b,t,wm=1.0,kern=True,wlm=1.0):\n    Lin=np.concatenate([LCc,wlm*(LM-LM.mean(0))],1)\n    R=np.concatenate([RC,wm*(LM-LM.mean(0))],1); rbar=np.linalg.norm(R,axis=1).mean()\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            if kern:\n                Rn=R[tr]/np.linalg.norm(R[tr],axis=1,keepdims=True)\n                Z=np.concatenate([wl*Lin,gelu(b*(R@Rn.T)/rbar-t)],1)\n            else: Z=Lin\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nprint('lin', cv([100,300,1000],1,0,0,kern=False))\nfor b,t in [(10,4.5),(10,5.5),(15,8)]:\n    for wl in [0.3]:\n        print(b,t,wl, cv([3,10,30],wl,b,t))"
}
```


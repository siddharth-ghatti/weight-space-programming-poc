## reasoning summary

Linear regression is doing worse here, and it looks like it needs a much larger ridge penalty because the raw features are on a large scale. At λ=30 it sits at 96.2, which is below the 97.48 I got earlier without centering at a ridge of 1000, so the raw geometry is clearly worse than the standardized version. Centering also makes the per-sample σ_u spread widely, from about 0.14 to 0.33, so per-sample normalization is changing the results a lot.

I'm weighing whether to keep the uncentered version, since it held up better at 97.48, versus normalizing the deviations, which discards the spread in magnitude. One option is to reserve a few dimensions for a constant anchor so the norm stays more stable.

I'm considering anchoring the norm with a large constant component in a few spare dimensions, which would make per-sample σ nearly constant and turn the layer norm into roughly a linear map. That would bring back the unnormalized dot-product kernel, and an earlier run of that kind reached 97.75. The open question is how much the lost cosine normalization matters, so I want to test it offline with the kernel built on the separated PCs, where the cosine version reached 97.95.

I'm checking whether pseudo-cos, which divides by the norm of the reference vector only, loses much accuracy compared with full cosine. If it does, then normalization is the key ingredient. I'm also considering normalizing through the GELU gate with a cosine term, but that only works if the whole vector is what gets normalized, and with the offset in the uncentered version the centered cosine isn't directly computable. What I care about in the end is cross-validated accuracy, so I'll set the test up around that.

I'm going to test pseudo-cos and other variants in the ideal Zs setting first, to see which ingredient actually matters before I design around it.



Centering u made results worse (~96–97%), so I'm checking whether the kernel truly needs cosine normalization or whether a pseudo-cosine that the FFN can compute without normalizing works just as well.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nZs=np.concatenate([(C9-C9.mean(0))/C9.std(0),(M9-M9.mean(0))/M9.std(0)],1)\nnr=np.linalg.norm(Zs,axis=1); print('||Zs|| pct',np.percentile(nr,[1,50,99]))\ndef cv(Lin,R,lams,b,t,wl,mode):\n    rbar=np.linalg.norm(R,axis=1).mean()\n    if mode=='cos': Rn=R/np.linalg.norm(R,axis=1,keepdims=True); G=Rn@Rn.T\n    elif mode=='pseudo': G=(R@R.T)/np.linalg.norm(R,axis=1)[None,:]/rbar\n    elif mode=='dot': G=(R@R.T)/rbar**2\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            Z=np.concatenate([wl*Lin,gelu(b*G[:,tr]-t)],1)\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nfor mode in ['cos','pseudo','dot']:\n    for b,t in [(10,4.5),(15,7)]:\n        print(mode,b,t,cv(Zs,Zs,[10,30],b,t,0.3,mode))"
}
```


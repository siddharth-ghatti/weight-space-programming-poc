## reasoning summary

Supervised LDA projection to 8 dimensions gets close to the full model's accuracy, so most of the gain seems to come from a small supervised stash. The open question is what happens if the stash is fit on all the data before the readout is re-fit on practice.

I'm checking whether the stash-then-readout setup is honest, and I think the existing CV already is: the LDA and the readout are both fit only on the training fold, so the 97.95% holds. I could instead keep the joint-ridge stash fixed without re-fitting, which would give 98.08%, and I'm weighing which option is simpler and more reliable.

I'm weighing two ways to use the stash: keep the joint-ridge weights fixed and approximate the stash dims in the readout, or refit the readout with the stash as features. The second one can be checked honestly in CV by fitting the joint ridge on the training fold and then refitting the readout on that fold's in-sample stash values, so I'll simulate that and compare it against the LDA version.

I'm going to check whether the noise in the scaling factor α matters, so I'll simulate it with a lognormal draw.



A supervised 8-dim stash recovers most of the M0 gain (~97.95–97.99% vs 97.64%), while unsupervised PCs don't help. Next I'll simulate the real pipeline with a joint-ridge stash and per-sentence α noise.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64); M0=F['M0'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef stdz(X,tr): m=X[tr].mean(0); s=X[tr].std(0)+1e-6; return (X-m)/s\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\ndef lda_stash(tr,kr=0.1):\n    X=M0; mu=X[tr].mean(0); Xs=X-mu\n    Mc=np.array([Xs[tr][y[tr]==c].mean(0) for c in range(8)])\n    R=Xs[tr]-Mc[y[tr]]; Sw=R.T@R/len(R)\n    W=np.linalg.solve(Sw+kr*np.trace(Sw)/768*np.eye(768),Mc.T); P=Xs@W; return P/P[tr].std(0)\ndef joint_stash(tr,lam=1000):\n    Z=np.concatenate([stdz(C9,tr),stdz(M9,tr),stdz(M0,tr)],1)\n    Wt,b=ridge_W(Z[tr],np.eye(8)[y[tr]]-1/8,lam)\n    P=Z[:,1536:]@Wt[1536:]; return P/P[tr].std(0)\nrng=np.random.default_rng(0)\ndef run(stash_fn, lams, w=1.0, noise=0.0, kern=None):\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            parts=[stdz(C9,tr),stdz(M9,tr)]\n            if stash_fn is not None:\n                S=stash_fn(tr); al=np.exp(noise*rng.standard_normal(len(y)))[:,None]; S=S*al\n                parts.append(w*stdz(S,tr))\n            X=np.concatenate(parts,1)\n            if kern is not None:\n                beta,theta,wl=kern\n                Xn=X/np.linalg.norm(X,axis=1,keepdims=True)\n                Kf=gelu(beta*(Xn@Xn[tr].T)-theta); X=np.concatenate([wl*X,Kf],1)\n            Wt,b=None,None\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams:\n                Wt,b=ridge_W(X[tr],Y,l); res[l]+=((X[te]@Wt+b).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nprint('joint stash w1', run(joint_stash,[300,1000],1))\nprint('joint stash w2', run(joint_stash,[300,1000],2))\nfor nz in [0.1,0.2,0.3]:\n    print('LDA stash noise',nz, run(lda_stash,[300,1000],1,nz))\nprint('kernel no stash', run(None,[10,30],kern=(10,4.5,0.3)))\nprint('kernel + LDA stash', run(lda_stash,[10,30],1,0.0,kern=(10,4.5,0.3)))\nprint('kernel + LDA stash noise .2', run(lda_stash,[10,30],1,0.2,kern=(10,4.5,0.3)))"
}
```


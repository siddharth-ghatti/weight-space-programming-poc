## reasoning summary

I'm merging the C7, C8, C9, M7, M8 and M9 stash version into featsS.npz, then moving on to the honest cross-validation. For each sample I'm fitting α and β from regressing C9'[S] on z = 4·st + h0 using the all-data stash, where the fit is exact, so C9'[S] equals α·z plus β.

I'm working out how to simulate the stash honestly in each fold: the joint ridge for the stash would be fit in-fold on the base features, as the real pipeline does, and then the simulated stash comes from the per-sample α and β applied to the fold's predicted stash. The readout would then be ridge on the kept C9 and M9 plus the weighted simulated stash, to be compared against the base [C9, M9] score of 97.64 and the new keep-only setup.



I'm merging the two reads into featsS.npz and running the honest CV, refitting the M0-stash weights on each training fold and rebuilding the stash values from the measured per-sample α and β.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\ncfg=np.load('stash_cfg.npz'); S=cfg['S']; A0=cfg['A0']; b0=cfg['b0']\nD={}\nfor l in [7,8,9]:\n    D[f'C{l}']=np.concatenate([np.load(f'pass_12/cls_hidden_L{l}.npy'),np.load(f'pass_13/cls_hidden_L{l}.npy')])\n    D[f'M{l}']=np.concatenate([np.load(f'pass_13/mean_hidden_L{l}.npy')[:0],np.load(f'pass_12/mean_hidden_L{l}.npy'),np.load(f'pass_13/mean_hidden_L{l}.npy')])\nnp.savez('featsS.npz',**D)\nC9n=D['C9'].astype(np.float64); M9n=D['M9'].astype(np.float64)\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64); M0=F['M0'].astype(np.float64)\nh0=F['C0'][0,S].astype(np.float64)\nst=M0@A0.T+b0; z=4*st+h0\nal=np.zeros(len(y)); be=np.zeros(len(y)); r2=np.zeros(len(y))\nfor i in range(len(y)):\n    X=np.c_[z[i],np.ones(8)]; c=np.linalg.lstsq(X,C9n[i,S],rcond=None)[0]; al[i],be[i]=c\n    r2[i]=1-((C9n[i,S]-X@c)**2).sum()/((C9n[i,S]-C9n[i,S].mean())**2).sum()\nprint('alpha',al.mean(),al.std()/al.mean(),'r2 min',r2.min())\nnp.savez('stash_alpha.npz',al=al,be=be)\nkeep=np.setdiff1d(np.arange(768),S)\ndef sz(X,tr): m=X[tr].mean(0); s=X[tr].std(0)+1e-9; return (X-m)/s\ndef ridge(Xtr,Y,lam,Xte):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return (Xte-xm)@Wt+ym, Wt\ndef stash_fold(tr,lam=1000):\n    Z=np.concatenate([sz(C9,tr),sz(M9,tr),sz(M0,tr)],1)\n    _,Wt=ridge(Z[tr],np.eye(8)[y[tr]]-1/8,lam,Z[:1])\n    s=Z[:,1536:]@Wt[1536:]; return s-s[tr].mean(0)\ndef cv(mode,lams,w=1.0):\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            if mode=='base': X=np.c_[sz(C9,tr),sz(M9,tr)]\n            elif mode=='newkeep': X=np.c_[sz(C9n[:,keep],tr),sz(M9n[:,keep],tr)]\n            elif mode=='newstash':\n                sf=stash_fold(tr); sim=al[:,None]*(4*sf+h0)+be[:,None]\n                X=np.c_[sz(C9n[:,keep],tr),sz(M9n[:,keep],tr),w*sz(sim,tr)]\n            elif mode=='basestash':\n                sf=stash_fold(tr); X=np.c_[sz(C9,tr),sz(M9,tr),w*sz(sf,tr)]\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams: res[l]+=(ridge(X[tr],Y,l,X[te])[0].argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nL=[300,1000]\nprint('base     ',cv('base',L)); print('newkeep  ',cv('newkeep',L))\nfor w in [1,2]: print('newstash w',w,cv('newstash',L,w)); \nprint('basestash',cv('basestash',L))"
}
```


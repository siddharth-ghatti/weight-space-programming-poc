## reasoning summary

Checking the honest cross-validation numbers first: the newstash model is around 98.0 against a baseline of 97.64, and perturbing the data costs essentially nothing on the full set. Next I'm setting up the layer 9 readout, using the stash-inclusive features and the keep-dimension mask, following the fit_readout design in readout.py.

Working out the readout: the channel dimensions have to come from the 8 lowest-variance dims of C9', but the stash dims have tiny variance and would get picked by mistake, so I'll exclude them from that selection. Then the keep weights go over the keep dims including the stash dims, and the mask A gets zeroed on the stash dims. The row-centering is safe since the weights sum to zero, and the per-sample LayerNorm scale doesn't change the argmax.

I'm checking the stash dimensions for numerical trouble: the stash coefficients will be large, but float32 precision should still be fine since the values are fit directly. I'm planning a custom fit that gives the stash block its own weight, with the readout converted back to keep-dimension weights afterward.

I'm deciding how the attention-side weights should treat the stash dimensions: zeroing them on the mean over tokens, though those dimensions still carry some token-embedding signal, so excluding them is a simplification I'm accepting for cleanliness.

I'm settling the hyperparameters with a small cross-validation grid on the honest simulation, using the same row-centered readout parametrization, to pick the channel weight, stash weight, and ridge penalty. Then I'll write the layer 9 readout with those choices.

Checking how the readout's output dimensions map back through the layer 9 weights, including whether the CLS's own channel values add a constant offset that the pooler has to account for.

I'm checking whether the layer 9 readout's constant offset needs correcting: the channel dimensions' mean shifts each output by a fixed amount that the bias has to absorb. I'll look at the saved readout_L9 file to see whether the earlier fit already included that correction, then confirm the layer 9 LayerNorm and FFN settings leave the later layers as identity.

I'm planning to check the in-sample logits computed offline against the saved readout parameters, then run a predict pass on a sample of the practice set as a correctness check, comparing argmax and correlation since the layer norm rescales the outputs.



The honest CV for the stash model is about 98.0%, against 97.64% for the current readout, and the stash edit itself costs nothing measurable. Next I'm tuning the readout hyperparameters under the same honest simulation.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\ncfg=np.load('stash_cfg.npz'); S=cfg['S']\nD=np.load('featsS.npz'); C9n=D['C9'].astype(np.float64); M9n=D['M9'].astype(np.float64)\nA=np.load('stash_alpha.npz'); al=A['al']; be=A['be']\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64); M0=F['M0'].astype(np.float64); h0=F['C0'][0,S].astype(np.float64)\nnonS=np.setdiff1d(np.arange(768),S)\nch=nonS[np.argsort(C9n[:,nonS].var(0))[:8]]; print('ch',ch)\nkeep=np.setdiff1d(np.arange(768),ch)            # includes S\nMcols=np.setdiff1d(np.arange(768),S)\ndef sz(X,tr): m=X[tr].mean(0); s=X[tr].std(0)+1e-9; return (X-m)/s\ndef ridge(Xtr,Y,lam,Xte):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return (Xte-xm)@Wt+ym, Wt\ndef stash_fold(tr,lam=1000):\n    Z=np.concatenate([sz(C9,tr),sz(M9,tr),sz(M0,tr)],1)\n    _,Wt=ridge(Z[tr],np.eye(8)[y[tr]]-1/8,lam,Z[:1]); s=Z[:,1536:]@Wt[1536:]; return s-s[tr].mean(0)\nSF=[[stash_fold(fo!=f) for f in range(5)] for fo in FOLDS]\ndef cv(wc,ws,lams):\n    res={l:0 for l in lams}\n    for r,fo in enumerate(FOLDS):\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            Csim=C9n.copy(); Csim[:,S]=al[:,None]*(4*SF[r][f]+h0)+be[:,None]\n            Ck=Csim[:,keep]; Cc=Ck-Ck.mean(1,keepdims=True)\n            isS=np.isin(keep,S)\n            Zc=sz(Cc,tr); Zc[:,~isS]*=wc; Zc[:,isS]*=ws\n            X=np.c_[Zc,sz(M9n[:,Mcols],tr)]\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams: res[l]+=(ridge(X[tr],Y,l,X[te])[0].argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nfor wc in [0.7,1.0]:\n    for ws in [1,2,3]:\n        print(wc,ws,cv(wc,ws,[300,500,1000]))"
}
```


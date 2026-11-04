## reasoning summary

I'm settling on the weights: the three candidate settings all land around 98.0–98.1, so the differences look like noise. I'll go with the middle option, wc=0.85, ws=1, λ=500. Next I'm fitting the final readout on all the data, using the real C9' with the actual stash and M9', and writing a custom fit procedure for it.

Now I'm writing the custom fit: centering the features, ridge-regressing the readout weights, and then shifting B so it applies correctly to the uncentered kept features.

The remaining step is to check that the centered-then-uncentered shift is exact: subtracting the mean of the readout weights leaves their product with the kept features unchanged, and the leftover offset is the same for every class, so the argmax is unaffected. The channel term only adds a tiny residual that can be ignored, which leaves the final logits matching the fit. Next I'll write `fit_readout2` in the run and save the result to `readout_S`.

I'm now writing out the file-saving step: the readout gets stored to a `.npz` file, then passed to `write_layer11_readout` with the scale and hch_mean settings. I'm checking that the pooler scale of s=1 still keeps the logits in the linear range of tanh, since the final layernorm standard deviation comes out around 5. I also want to confirm that A is placed correctly on the 768 M9 dimensions with zero columns at S.

Checking the file-saving step now.



All settings land around 98.0–98.1%, so the differences are noise. I'm fitting the final readout on all 1496 samples with wc=0.85, ws=1, λ=500, then converting it to pooler and attention weights at layer 9 and checking the in-sample logits.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys\nsys.path.insert(0,'.'); from readout import write_layer11_readout\nF=np.load('feats.npz'); y=F['y']\ncfg=np.load('stash_cfg.npz'); S=cfg['S']\nD=np.load('featsS.npz'); C=D['C9'].astype(np.float64); M=D['M9'].astype(np.float64)\nnonS=np.setdiff1d(np.arange(768),S)\nch=nonS[np.argsort(C[:,nonS].var(0))[:8]]\nkeep=np.setdiff1d(np.arange(768),ch); Mcols=np.setdiff1d(np.arange(768),S)\nwc,ws,lam=0.85,1.0,500.0\nCk=C[:,keep]; Cc=Ck-Ck.mean(1,keepdims=True)\nmuC=Cc.mean(0); sdC=Cc.std(0)+1e-9; muM=M[:,Mcols].mean(0); sdM=M[:,Mcols].std(0)+1e-9\nwv=np.where(np.isin(keep,S),ws,wc)\nZ=np.c_[wv*(Cc-muC)/sdC,(M[:,Mcols]-muM)/sdM]\nY=np.eye(8)[y]-1/8; zm=Z.mean(0); Zc=Z-zm\na=np.linalg.solve(Zc@Zc.T+lam*np.eye(len(Z)),Y-Y.mean(0)); Wt=Zc.T@a\nnk=len(keep); Wc=Wt[:nk]; Wm=Wt[nk:]\nB=(wv[:,None]*Wc/sdC[:,None]).T\nA=np.zeros((8,768)); A[:,Mcols]=(Wm/sdM[:,None]).T\nb=Y.mean(0)-zm@Wt-(wv*muC/sdC)@Wc-(muM/sdM)@Wm\nB=B-B.mean(1,keepdims=True)\nlogit=Ck@B.T+M@A.T+b\nref=Z@Wt+(Y.mean(0)-zm@Wt)\nprint('max diff vs ridge',np.abs(logit-ref).max(),'in-sample acc',(logit.argmax(1)==y).mean())\nR=dict(B=B,A=A,b=b,ch=ch,keep=keep)\nnp.savez('readout_S.npz',**R)\nhm=C[:,ch].mean(0)\nwrite_layer11_readout(R, save_edit, c=100.0, s=1.0, hch_mean=hm, layer=9)\nnp.save('offline_logits_S.npy',logit)\nprint('B abs max',np.abs(B).max(),'A abs max',np.abs(A).max(), 'logit std',logit.std())"
}
```


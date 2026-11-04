## reasoning summary

Results hold steady around 98.0 across seeds 10 through 14 at 10-fold, so the hyperparameters aren't the issue. I'm also weighing the CLS term in the pooling step, though with c between 10 and 20 it adds little. Anchor selection is chosen per fold from the training data in cross-validation, while the final model uses all of it, and I'm now deciding whether to pull in more deployment information.

I'm checking whether the deployed linear carry direction, fit on all the data, actually lines up with a fresh joint fit. The leaky 99.8 CV number worries me, since it suggests the carry direction is overfitting the practice set.

I'm checking whether the carry direction that was fit on all the data still holds up as the honest generalization estimate, since the joint fit's CV already covers that. What I still need to know is which kernel the carry weights were fit with, because if it was a different one, refitting the pooling on the new kernel would let the linear carry dominate. I'll look at the magnitude of pool1's linear coefficients against the anchor terms to see how much it's leaning on the carry.

I'm thinking the deployed model should match the joint fit on all 1496 samples that CV evaluates, so the carry direction would be the 760x8 linear map from that fit. The per-sample scale noise in the carry is the catch, and refitting the pool on measured carry scores is how the approach absorbs it. What I can't do yet is measure the carry scores that would make that refit exact.

I'm working out whether to predict the new carry's pooled values from the current measurements, scaling each sample's linear map by a per-sample factor, since the layer norms should leave that scale roughly unchanged. Before committing to that, I want to see how far the current carry direction actually differs from the new joint fit.

I'm going to check whether the carry weights from the joint fit on all 1496 samples match the deployed ones, comparing the 760x8 linear block directly. I also want to see how large pool1's linear coefficients are relative to the anchor terms.



The grid is flat at about 98.0 across 10-fold CV, so tuning won't add margin. The bigger risk is that CV refits the 760-dim linear block per fold while deployment only rescales the fixed 8-dim wlin carry, so I'm measuring on the full data how much of the new linear block falls outside span(wlin).



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib7 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS); Y=np.eye(8)[y]\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\ncp=np.load('carry_prov.npz'); wlin=cp['wlin']; wraw=cp['wraw']; mu0=cp['mu0']\ntr=np.ones(1496,bool)\ndef jointfit(M,C,M0,c=10,lam=.05,c0=.5,na=1008):\n    mu=M.mean(0); Z=pool_z(M,C,c,mu); Wf=anchor_dirs(Z); Kc=gelu(Z@Wf.T); Kc=.5*(Kc+Kc.T)\n    piv=pivchol(Kc,na); Wd=Wf[piv]; Kaa=Kc[np.ix_(piv,piv)]; Phi=feats(Z,Wd); Lin=lin_feats(M0,tr)\n    coef,b=fit(Phi,Lin,Y,Kaa,lam,c0); return coef,b,Phi,Lin,piv,Z\ncoef,b,Phi,Lin,piv,Z=jointfit(M,C,M0)\nWn=coef[1008:]; \nSl=Lin@Wn; So=Lin@wlin\nB,_,_,_=np.linalg.lstsq(np.c_[So,np.ones(1496)],Sl,rcond=None)\nres=Sl-np.c_[So,np.ones(1496)]@B\nprint('R2 new lin scores from old carry span', 1-(res**2).sum()/((Sl-Sl.mean(0))**2).sum())\nprint('norm of lin part vs kernel part', np.std(Sl), np.std(Phi@coef[:1008]))\nS=np.c_[Phi,Lin]@coef+b; print('train acc joint', (S.argmax(1)==y).mean())\n# how was wlin made? compare to joint fit on old seq1024 / base features\nfo=np.load('feats_seq1024_depth.npz'); \ncoef2,_,_,_,_,_=jointfit(fo['M9'].astype(np.float64),fo['C9'].astype(np.float64),M0)\nS2=Lin@coef2[1008:]; B2,_,_,_=np.linalg.lstsq(np.c_[So,np.ones(1496)],S2,rcond=None)\nr2=S2-np.c_[So,np.ones(1496)]@B2; print('R2 seq1024 joint lin from carry span',1-(r2**2).sum()/((S2-S2.mean(0))**2).sum())\nprint('wlin vs coef2 lin corr', np.corrcoef(wlin.ravel(), coef2[1008:].ravel())[0,1], 'vs Wn', np.corrcoef(wlin.ravel(), Wn.ravel())[0,1])"
}
```


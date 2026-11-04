## reasoning summary

Both settings come out almost the same at 20-fold, around 98.2 to 98.3, so I'll go with drop46 for deployment. I'm now checking whether c=20/cr=0 holds up against c=10/cr=1 for drop46.

Settling on c=20 with cr=0 for deployment. Its residual-dimension handling is consistent with the simulation, and the carried-over residual noise in C[RES] stays realistic. Next I'm doing the final joint fit on all 1496 samples and mapping the linear weights back to the raw feature space.

I'm centering the new weight vector by subtracting its mean across the eight residual columns, since a common offset is class-invariant and only shifts layer-norm statistics. The simulation fit its model on centered vectors, so the final fit should match that, and the scaling factor was computed on the uncentered version, which is a minor difference.

I'm checking the deployed scores next: combining the feature-based term, the linear term on the residual columns, and the bias, then comparing training accuracy against the joint fit.

Next I'm installing the final weights: the L0 value projection gets the centered residual weights on the kept dimensions, with the bias set to offset the mean, and the L11 output projection keeps the feature dimensions at full strength while zeroing the residual ones.

Setting the attention output's residual diagonal to zero and its bias to zero, so the residual slots just carry the CLS values through unchanged. The rest of the layer-norm and feed-forward setup is already in place, so I'm working through the remaining FFN weight rows.

Still working through the residual weights for the feed-forward layer, checking how the pair units combine with the centered carry term.

I'm checking whether the layer 11 output weights for the kept units are really zero, since the earlier edit changed 8192 entries, which suggests only the residual rows were touched.

I'm now picking the scale factor for the final output weights, aiming for tanh inputs that stay moderate so that the argmax is preserved.

Checking whether the pooled CLS residual is still realistically noisy, since its unexplained part makes the carry term less than perfectly predictable. I'm also working out how large the linear weight on the residual pair units needs to be, given that the residual values are small.

I'm planning the verification step: a read pass on the edited model for the first 1024 samples, comparing its logits and layer-7 features against my numpy predictions. After that I'll run the remaining samples. Before installing anything, I'll back up the current layer 0 and layer 11 edits.



At 20-fold the deployed versions tie (98.28 vs 98.24), so I'm dropping layer 4 and keeping the smaller structure with 1008 anchors. Next I'll run the final fit on all 1496 samples and check the scale and saturation before installing anything.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib9 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS); Y=np.eye(8)[y]\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\ncp=np.load('carry_prov.npz'); wraw=cp['wraw'].astype(np.float64); mu0=cp['mu0'].astype(np.float64)\ncold=(M0-mu0)@wraw; gM=carry_gain(M[:,RES],cold); gC=carry_gain(C[:,RES],cold)\ntr=np.ones(1496,bool); c=20; cr=0.0; lam=.05; c0=.5\nLin=lin_feats(M0,tr); muK=M0[:,KEEP].mean(0); sn=np.sqrt(((M0[:,KEEP]-muK)**2).sum(1).mean())\nprint('lin_feats check', np.abs(Lin-(M0[:,KEEP]-muK)/sn).max())\nmu=M.mean(0); Z=pool_z(M,C,c,mu,cr); Wd,Kaa,piv=prep_anchors(Z,tr)\nPhi=feats(Z,Wd); coef,b=fit(Phi,Lin,Y,Kaa,lam,c0)\nWl=coef[1008:]; Wl_c=Wl-Wl.mean(1,keepdims=True)\nSl=Lin@Wl_c\nlamb=np.sqrt((cold**2).sum()/(Sl**2).sum())\nwnew=np.zeros((768,8)); wnew[KEEP]=lamb*Wl_c/sn\ncnew=(M0-mu0)@wnew; print('cnew vs lamb*Sl', np.abs(cnew-lamb*Sl).max(), 'std old/new', cold.std(0).round(3), cnew.std(0).round(3))\nMn=M.copy(); Cn=C.copy(); Mn[:,RES]+=gM[:,None]*(cnew-cold); Cn[:,RES]+=gC[:,None]*(cnew-cold)\nmun=Mn.mean(0); Zn=pool_z(Mn,Cn,c,mun,cr); Phin=feats(Zn,Wd)\nprint('Z KEEP change', np.abs(Zn[:,KEEP]-Z[:,KEEP]).max(), 'Phi change', np.abs(Phin-Phi).max())\nL8=Zn[:,RES]-Zn[:,RES].mean(1,keepdims=True); Slc=Sl\nbeta=(L8*Slc).sum()/(L8**2).sum(); print('beta',beta, 'R2 L8->Sl', 1-((beta*L8-Slc)**2).sum()/(Slc**2).sum())\nSj=np.c_[Phi,Lin]@coef+b; Sk=Phin@coef[:1008]\nSdep=Sk+beta*L8+b\ncen=lambda S:S-S.mean(1,keepdims=True)\nprint('train acc joint',(Sj.argmax(1)==y).mean(),'dep',(Sdep.argmax(1)==y).mean(), 'agree', (Sj.argmax(1)==Sdep.argmax(1)).mean())\nsrt=np.sort(Sdep,1); print('margin min/5pct', (srt[:,-1]-srt[:,-2]).min().round(4), np.percentile(srt[:,-1]-srt[:,-2],5).round(4))\nprint('centered score range', cen(Sdep).min().round(3), cen(Sdep).max().round(3), 'Zn RES std', Zn[:,RES].std().round(4))\nnp.savez('final_design.npz', coef=coef, b=b, Wd=Wd, piv=piv, wnew=wnew, mu0=mu0, mun=mun, mu=mu, beta=beta, lamb=lamb, c=c, cr=cr, Sdep=Sdep, Sj=Sj, Mn=Mn, Cn=Cn, gM=gM, gC=gC)\nwith open('notes.md','a') as fh: fh.write(\"- KEY: 2-stage (fixed carry dir + refit 8x8 on train) loses ~0.25% (stacking overfit). Deploy joint fit EXACTLY: new L0 carry = joint linear block (centered, magnitude-matched), pool linear weight beta fixed. Faithful deploy sim 20f: drop46 98.28, drop6skip 98.24 => keep drop46.\\n- final pool: c=20, cr=0 (RES diag 0), lam=.05 c0=.5 na=1008. Design in final_design.npz, lib9.py (deploy_sim, carry_gain, prep_anchors).\\n\")"
}
```


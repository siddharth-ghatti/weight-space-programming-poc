## reasoning summary

Cold seems to have no common-mode offset across classes, since its predictions barely vary in mean, which fits the linear block being centered. Then the residual energy in pool x at c=10 turns out to be about 20% of the total, so it's not negligible, and with M[RES] sitting around 0.4 with spread near 1, the carry looks substantial.

I'm now predicting the new RES values for M and C using the hidden-level model, then recomputing z at c=10. Next I'd like to test the deployed predictions with a cross-validation style simulation, refitting per fold so the carry-related normalization is handled consistently.

I'm weighing a leaner setup: shrink the RES contribution with cr=0.1 so the pool normalization barely depends on the carry, then keep c=10 and read the linear term from the scaled RES dims, with β compensating. That way the kernel features stay consistent between the old and new carry.

I'm still on the residual pass-through: the carry in the RES dims seems to matter for the pool normalization, but the CLS residual adds into the stream at full weight, so scaling the attention contribution alone won't remove it.

I'm estimating the energy split between the carry and the keep dimensions at c=10, where the CLS contribution to the RES dims is roughly 2.6 and the rest depends on how large the keep term is.

I'm weighing whether to keep the carry at full strength, since with cr=0 the RES energy would drop to about 6% but the M[RES] signal is the better read of the linear term. I'm leaning toward keeping cr=1 and c=10, then running the faithful simulation with predicted new M and C values to fit coefficients on the updated features.

I'm considering a two-pass fit: first fit the joint model on the current features, use it to predict the new carry values for M and C, then recompute the features from those predictions and refit the coefficients.

Pinning down the linear term now: if I make the pool features independent of the carry by zeroing its contribution from M[RES], the RES share falls to roughly 1.6% at c=20, but the C[RES] part still carries the CLS signal. That makes the linear read noisier, since the CLS residual brings in embedding offsets I don't want.

I'm checking whether the per-sample scale noise from the RES normalization is harmless, since the simulation with a 15% scale variation already matched the joint fit. The direction stays fixed, so what really matters is the overall weight β.

I'm setting up a faithful simulation for each fold: refit the features on the training split, predict the new carry values, and compare that against the joint CV result.

I'm setting the faithful simulation's steps: refit the features on the training split, shift the linear term into a new carry estimate, and derive the new M and C values from it.

I'm setting up the faithful simulation's last steps: recompute the pool features from the predicted new carry values, center the linear term across classes, and fit its scale against the carry-based term on the training split. Then I'll score the test split using the coefficients fixed from the first pass.

I'm treating the carry-free option as a fallback: a faithful simulation per fold, with the kernel features recomputed from the predicted new carry, and I'll measure whether the same setup holds up.



The carry columns sum to zero across classes, and RES makes up 20% of the pool input's energy at c=10, so changing the carry also shifts the pool normalisation. I'm running a faithful CV simulation of the deployment, comparing c=10 with cr=1 against c=20 with cr=0.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\ncode = r'''\nimport numpy as np, json\nfrom lib7 import *\ndef carry_gain(X, cold):\n    Xc=X-X.mean(1,keepdims=True); pc=cold-cold.mean(1,keepdims=True)\n    a=(Xc*pc).sum()/(pc**2).sum(); k=(Xc*pc).sum(1)/(pc**2).sum(1)/a\n    return a*k\ndef prep_anchors(Z, tr, na=1008):\n    Zt=Z[tr]; Wf=anchor_dirs(Zt); Kc=gelu(Zt@Wf.T); Kc=.5*(Kc+Kc.T)\n    piv=pivchol(Kc,na); return Wf[piv], Kc[np.ix_(piv,piv)], piv\ndef deploy_sim(M,C,M0,y,tr,te,cold,gM,gC,c=10,cr=1.0,lam=.05,c0=.5,refit=False):\n    Y=np.eye(8)[y]; Lin=lin_feats(M0,tr); mu=M[tr].mean(0)\n    Z=pool_z(M,C,c,mu,cr); Wd,Kaa,piv=prep_anchors(Z,tr)\n    Phi=feats(Z,Wd); coef,b=fit(Phi[tr],Lin[tr],Y[tr],Kaa,lam,c0)\n    Sl=Lin@coef[1008:]\n    lamb=np.sqrt((cold[tr]**2).sum()/(Sl[tr]**2).sum()); cnew=lamb*Sl\n    Mn=M.copy(); Cn=C.copy(); Mn[:,RES]+=gM[:,None]*(cnew-cold); Cn[:,RES]+=gC[:,None]*(cnew-cold)\n    mun=Mn[tr].mean(0); Zn=pool_z(Mn,Cn,c,mun,cr); Phin=feats(Zn,Wd)\n    L8=Zn[:,RES]-Zn[:,RES].mean(1,keepdims=True); Slc=Sl-Sl.mean(1,keepdims=True)\n    beta=(L8[tr]*Slc[tr]).sum()/(L8[tr]**2).sum()\n    Sjoint=np.c_[Phi,Lin]@coef+b\n    Sdep=Phin@coef[:1008]+beta*L8+b\n    out=dict(joint=(Sjoint[te].argmax(1)==y[te]).mean(), dep=(Sdep[te].argmax(1)==y[te]).mean())\n    if refit:\n        c2,b2=fit(Phin[tr],np.zeros((len(y),0)),Y[tr]-beta*L8[tr],Kaa,lam,c0)\n        S2=Phin@c2+beta*L8+b2; out['dep_refit']=(S2[te].argmax(1)==y[te]).mean()\n    return out, dict(coef=coef,b=b,Wd=Wd,piv=piv,lamb=lamb,beta=beta,mu=mu,mun=mun,Mn=Mn,Cn=Cn,Sdep=Sdep,Sjoint=Sjoint,Zn=Zn)\n'''\nopen('lib9.py','w').write(code)\nimport numpy as np, json\nfrom lib9 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS)\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\ncp=np.load('carry_prov.npz'); wraw=cp['wraw'].astype(np.float64); mu0=cp['mu0'].astype(np.float64)\ncold=(M0-mu0)@wraw; gM=carry_gain(M[:,RES],cold); gC=carry_gain(C[:,RES],cold)\nfor (c,cr) in [(10,1.0),(20,0.0)]:\n    R=[]\n    for s in range(10,14):\n        fo_=folds(y,10,s); acc={'joint':0,'dep':0,'dep_refit':0}\n        for i in range(10):\n            tr=fo_!=i; te=~tr\n            o,_=deploy_sim(M,C,M0,y,tr,te,cold,gM,gC,c=c,cr=cr,refit=True)\n            for kk in o: acc[kk]+=o[kk]*te.sum()\n        R.append({kk:v/1496 for kk,v in acc.items()})\n    print(c,cr,{kk:(np.round([r[kk] for r in R],4).tolist(), round(np.mean([r[kk] for r in R]),4)) for kk in R[0]},flush=True)"
}
```


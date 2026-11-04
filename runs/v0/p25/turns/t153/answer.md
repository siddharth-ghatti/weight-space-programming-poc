## reasoning summary

I'm weighing whether the 370 centers I drop should be picked at random or by a criterion like greedy pivoted Cholesky on the kernel matrix, since the scores I'm comparing are close.

I'm now testing center-selection strategies in cross-validation: for the mine centers, either random selection or pivoted Cholesky on the kernel residual, keeping the practice training points as centers too. The CV setup gives 2773 centers versus 3072 in the final model, which is acceptable for comparison. I also want to retune lambda and the weights on the new data, since the remaining errors partly reflect label noise and the gains are modest.



Adding my5 helps (CV 0.9924 → 0.9942, or 0.9951 as centers), but with 3442 points and only 3072 neurons I must choose 1576 of the 1946 sentences as centers. Next I'll compare random selection with pivoted Cholesky in CV, using the same number of my5 sentences as centers.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom kh import *\nD=load(); d5=np.load('my5_feats_clean.npz')\nZp,yp=D['Zp'],D['yp']\nZmine=np.concatenate([D['Zm'],D['Z3'],D['Z4'],d5['Z']]); ymine=np.concatenate([D['ym'],D['y3'],D['y4'],d5['y']])\nprint(len(Zmine))\nNC=1576\ndef pivchol_select(Zc_fixed, Zcand, k):\n    # greedy select candidates maximizing residual variance given fixed centers (kernel feature space, approx with gelu kernel)\n    allZ=np.concatenate([Zc_fixed,Zcand]); K=gelu(allZ@allZ.T/768)\n    nF=len(Zc_fixed); n=len(allZ)\n    # Cholesky residual diag after conditioning on fixed set\n    Kff=K[:nF,:nF]+1e-6*np.eye(nF); Kfc=K[:nF,nF:]\n    Lf=np.linalg.cholesky(Kff); V=np.linalg.solve(Lf,Kfc)  # nF x m\n    Kcc=K[nF:,nF:]-V.T@V\n    d=np.diag(Kcc).copy(); m=len(Zcand); sel=[]; G=np.zeros((k,m))\n    for j in range(k):\n        i=int(np.argmax(d)); sel.append(i)\n        g=(Kcc[i]-G[:j,i]@G[:j])/np.sqrt(max(d[i],1e-12)); G[j]=g; d=d-g**2; d[sel]=-np.inf\n    return np.array(sel)\ndef cv_sel(mode, wx=0.4, lam=0.007, seeds=(0,1,2), nf=5):\n    accs=[]\n    for s in seeds:\n        fo=folds(yp,s,nf); pred=np.zeros(1496,int)\n        for f in range(nf):\n            tr=fo!=f\n            if mode=='random': sel=np.random.default_rng(100+s).permutation(len(Zmine))[:NC]\n            elif mode=='piv': sel=pivchol_select(Zp[tr],Zmine,NC)\n            elif mode=='all': sel=np.arange(len(Zmine))\n            Cc=np.concatenate([Zp[tr],Zmine[sel]])\n            Zt=np.concatenate([Zp[tr],Zmine]); yt=np.concatenate([yp[tr],ymine]); wt=np.concatenate([np.ones(tr.sum()),np.full(len(Zmine),wx)])\n            a,b=fit(gelu(Zt@Cc.T/768),yt,wt,lam)\n            pred[~tr]=np.argmax(gelu(Zp[~tr]@Cc.T/768)@a+b,1)\n        accs.append((pred==yp).mean())\n    return round(np.mean(accs),4), np.round(accs,4)\nimport time; t=time.time()\nprint('random', cv_sel('random'), time.time()-t)\nprint('piv', cv_sel('piv'), time.time()-t)\n"
}
```


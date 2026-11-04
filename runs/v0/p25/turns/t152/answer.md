## reasoning summary

I'm checking the v4 model's accuracy on my own test sentences, which gets 96.3%. My sentences may follow a different distribution, and a few labels look ambiguous, like the stolen-cash deductible line and the renewal and grace-period premium lines.

I'm going to drop the ambiguous labels from the test set, like the stolen-cash deductible line, the grace-period lapse line, the renewal line, and the workers' comp payroll line, plus the extended reporting period endorsement. I'll keep the dependent coverage premium line and the employer notice line. After that I'll compare the non-center and center results on my5.



v4 gets 96.3% of the new sentences right; I'm dropping five ambiguous ones and then running CV with my5 added, first as centers and then as non-center training points.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom kh import *\nD=load()\nd5=np.load('my5_feats.npz'); Z5,y5,t5=d5['Z'],d5['y'],list(d5['texts'])\ndropk=['Stolen cash is only covered up to $200','If premium is not paid by the end of the grace period','We will renew your plan each year at the premium','Workers\\' comp premiums are based on accurate payroll','Payment for the extended reporting period endorsement']\nkeep=np.array([not any(t.startswith(s) for s in dropk) for t in t5]); print(keep.sum())\nZ5,y5=Z5[keep],y5[keep]; t5=[t for t,k_ in zip(t5,keep) if k_]\nnp.savez('my5_feats_clean.npz',Z=Z5,y=y5,texts=np.array(t5))\nZp,yp=D['Zp'],D['yp']\ndef cv_ny(extra_c, extra_nc, wx, lam, seeds=(0,1,2), nf=5):\n    accs=[]\n    for s in seeds:\n        fo=folds(yp,s,nf); pred=np.zeros(1496,int)\n        for f in range(nf):\n            tr=fo!=f\n            Cc=np.concatenate([Zp[tr]]+[e[0] for e in extra_c])\n            Zt=np.concatenate([Cc]+[e[0] for e in extra_nc]); yt=np.concatenate([yp[tr]]+[e[1] for e in extra_c+extra_nc])\n            wt=np.concatenate([np.ones(tr.sum()), np.full(len(Zt)-tr.sum(), wx)])\n            a,b=fit(gelu(Zt@Cc.T/768),yt,wt,lam)\n            pred[~tr]=np.argmax(gelu(Zp[~tr]@Cc.T/768)@a+b,1)\n        accs.append((pred==yp).mean())\n    return round(np.mean(accs),4), np.round(accs,4)\nm=(D['Zm'],D['ym']); m3=(D['Z3'],D['y3']); m4=(D['Z4'],D['y4']); m5=(Z5,y5)\nprint('v4 config', cv_ny([m,m3,m4],[],0.4,0.007))\nprint('m5 centers too', cv_ny([m,m3,m4,m5],[],0.4,0.007))\nprint('m5 non-centers', cv_ny([m,m3,m4],[m5],0.4,0.007))\n"
}
```


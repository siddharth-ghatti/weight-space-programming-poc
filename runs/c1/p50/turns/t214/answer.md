## reasoning summary

I'm saving the combined features to feats_drop46.npz first. After that I'll run full-1496 cross-validation across several seeds and a couple of hyperparameter settings, and compare it against the drop6skip full features (feats_drop6skip M8/C8).



## message

Pass 45 succeeded. I'll save the combined drop4+6 features, then run full-1496 10-fold CV (several seeds) side by side with drop6skip, using both regularisation settings.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib7 import *\nassert json.load(open('pass_45/meta.json'))['ids']==[r['id'] for r in practice()][1024:]\nM=np.concatenate([np.load('pass_44/mean_hidden_L7.npy'),np.load('pass_45/mean_hidden_L7.npy')]).astype(np.float64)\nC=np.concatenate([np.load('pass_44/cls_hidden_L7.npy'),np.load('pass_45/cls_hidden_L7.npy')]).astype(np.float64)\nd=np.load('feats_drop6skip.npz'); M0=d['M0'].astype(np.float64)\nnp.savez('feats_drop46.npz', M=M, C=C, M0=M0)\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS)\nwith open('notes.md','a') as fh: fh.write(\"\\n## Run 6\\n- drop46 subset 5 seeds: 98.01 (c10 .02 .5) vs drop6skip 97.99 => drop4 free. L7 skip->h9 err .196 > h8-as-proxy .175 => drop8 not viable. STOP pruning; finalize [0,1,2,3,5,7,8,11].\\n- pass_45 = rest472 L7 cls/mean -> feats_drop46.npz (M,C,M0 full 1496).\\n- pool reconstruction: c=20 (RES 1), mu=mean all, anchors units0-1007, lin pairs 1008-1023 on RES, pooler 2.106*centering on RES, cls=I.\\n\")\nimport time; t=time.time()\nfor (c,lam,c0) in [(10,.02,.5),(10,.05,.5),(5,.05,.25)]:\n    r=[cv(M,C,M0,y,c=c,lam=lam,c0=c0,seed=s) for s in range(4)]\n    print('drop46',c,lam,c0,np.round(r,4),round(np.mean(r),4), round(time.time()-t))"
}
```


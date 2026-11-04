## reasoning summary

I ran a diagnostic on 64 ids, listing them by hand, which is fine. Next I'm checking the stash dimensions at the CLS token in each layer against the intended stash, and fitting a per-sample regression of those values on the stash with a constant term to estimate α. I also want to see how the non-stash dimensions move relative to the base features.



I ran a 64-text diagnostic and I'm now checking how well the stash carries through each layer's CLS, plus how far the non-stash states drifted from base.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nF=np.load('feats.npz'); cfg=np.load('stash_cfg.npz'); S=cfg['S']; A0=cfg['A0']; b0=cfg['b0']\nids=json.load(open('all_ids.json')); idx=np.arange(64)   # p0000..p0063 assumed rows 0..63 of feats\nmeta=json.load(open('pass_11/meta.json')); P=practice(); \nprint(meta.keys() if isinstance(meta,dict) else None)\ntexts=[p['text'] for p in P]; assert all(texts[i]==meta['texts'][i] for i in range(64))\nM0=F['M0'][:64].astype(np.float64); st=M0@A0.T+b0\nkeep=np.setdiff1d(np.arange(768),S)\nfor l in [1,2,3,5,7,9]:\n    C=np.load(f'pass_11/cls_hidden_L{l}.npy').astype(np.float64); M=np.load(f'pass_11/mean_hidden_L{l}.npy').astype(np.float64)\n    Cb=F[f'C{l}'][:64].astype(np.float64); Mb=F[f'M{l}'][:64].astype(np.float64)\n    # per-sample fit C[S] = a*st + b\n    al=[];r2=[]\n    for i in range(64):\n        X=np.c_[st[i],np.ones(8)]; coef,res,_,_=np.linalg.lstsq(X,C[i,S],rcond=None); al.append(coef[0])\n        pred=X@coef; r2.append(1-((C[i,S]-pred)**2).sum()/((C[i,S]-C[i,S].mean())**2).sum())\n    al=np.array(al)\n    dC=np.linalg.norm(C[:,keep]-Cb[:,keep],axis=1)/np.linalg.norm(Cb[:,keep]-Cb[:,keep].mean(0),axis=1)\n    dM=np.linalg.norm(M[:,keep]-Mb[:,keep],axis=1)/np.linalg.norm(Mb[:,keep]-Mb[:,keep].mean(0),axis=1)\n    print(f'L{l}: alpha mean {al.mean():.3f} cv {al.std()/al.mean():.3f}  R2 med {np.median(r2):.4f}  |C[S]| {np.abs(C[:,S]).mean():.3f}  relchange C {dC.mean():.3f} M {dM.mean():.3f}')"
}
```


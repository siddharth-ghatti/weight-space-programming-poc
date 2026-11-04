## reasoning summary

I can run a partial CV check on the 1024 subset first: compare the base configuration against the new one, with the stash applied. If the new version is clearly worse, I'll stop there and skip reading the remaining 472.



Next I'll compare base and stash-edited models on the 1024-sentence subset, with and without stash dims, before deciding whether to spend a pass on the remaining 472 ids.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y'][:1024]; FOLDS=np.load('folds_rep.npy')[:,:1024]\ncfg=np.load('stash_cfg.npz'); S=cfg['S']\nkeep=np.setdiff1d(np.arange(768),S)\nC9n=np.load('pass_12/cls_hidden_L9.npy').astype(np.float64); M9n=np.load('pass_12/mean_hidden_L9.npy').astype(np.float64)\nC8n=np.load('pass_12/cls_hidden_L8.npy').astype(np.float64); M8n=np.load('pass_12/mean_hidden_L8.npy').astype(np.float64)\nC9=F['C9'][:1024].astype(np.float64); M9=F['M9'][:1024].astype(np.float64); M0=F['M0'][:1024].astype(np.float64)\ndef sz(X): return (X-X.mean(0))/(X.std(0)+1e-9)\ndef cv(blocks,lams):\n    X=np.concatenate([w*sz(B) for B,w in blocks],1); K=X@X.T\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8; ym=Y.mean(0); n=tr.sum()\n            Ktr=K[np.ix_(tr,tr)]; H=np.eye(n)-1/n; Kc=H@Ktr@H\n            Kte=K[np.ix_(te,tr)]; Kte=Kte-Kte.mean(1,keepdims=True)-Ktr.mean(0)[None]+Ktr.mean()\n            for l in lams:\n                a=np.linalg.solve(Kc+l*np.eye(n),Y-ym); res[l]+=((Kte@a+ym).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nL=[300,1000]\nprint('base C9M9      ', cv([(C9,1),(M9,1)],L))\nprint('base C9M9+M0   ', cv([(C9,1),(M9,1),(M0,1)],L))\nprint('new keep only  ', cv([(C9n[:,keep],1),(M9n[:,keep],1)],L))\nfor w in [1,2,4]:\n    print('new +stash w',w, cv([(C9n[:,keep],1),(M9n[:,keep],1),(C9n[:,S],w)],L))\nprint('new +stash +M0 ', cv([(C9n[:,keep],1),(M9n[:,keep],1),(C9n[:,S],2),(M0,1)],L))\nprint('new L8 +stash  ', cv([(C8n[:,keep],1),(M8n[:,keep],1),(C8n[:,S],2)],L))"
}
```


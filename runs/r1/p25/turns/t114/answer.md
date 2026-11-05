## reasoning summary

I'm weighing whether the remaining passes could cut risk at m=1792, though an ensemble-style approach doesn't seem right. I'm also wondering whether L5 could beat the full FFN, and whether a head correction with a CV-chosen lambda would help, since the earlier gain was small.

I'm checking whether the head correction holds up: the CV gains look small but consistent, so I'll test the corrected h6 mapping on the honest 1792 logits and only apply it if the gain persists across folds.



Before the verification pass, I'm testing a ridge-regularised correction of classifier.dense, which gave small gains in both earlier builds, and I'll keep it only if cross-validation confirms the gain here.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lsbuild import gelu, layer_data\nfrom evalh import Lb, lab1, f1, stats\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\nab,W1,b1,U,b2=layer_data(5,base)\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\nD=base('classifier.dense.weight').astype(np.float64); bd=base('classifier.dense.bias').astype(np.float64); O=base('classifier.out_proj.weight').astype(np.float64); bo=base('classifier.out_proj.bias').astype(np.float64)\nae=np.load('a5e_1792.npy').astype(np.float64); Ge=gelu(ae@W1.T+b1)\nkeep=json.load(open('keep_F1792.json')); k5=keep['5']\nWl=W(pre+'output.dense.weight').astype(np.float64); bl=W(pre+'output.dense.bias').astype(np.float64)\nhe=LN(ae+Ge[:,k5]@Wl[:,k5].T+bl,g2,bb)\nhb=np.load('pass_1/cls_hidden_L6.npy').astype(np.float64); Zb=hb@D.T+bd\nfolds=np.load('folds.npy'); pb=Lb.argmax(1); srt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]\nL0=np.tanh(he@D.T+bd)@O.T+bo\nprint('no corr: agree',(L0.argmax(1)==pb).mean())\nfor lam in [3e2,1e3,3e3,1e4]:\n  for seed in [0,1,2]:\n    if seed==0: fo=folds\n    else: fo=np.random.RandomState(seed).permutation(1000)%10\n    Lcv=np.zeros_like(Lb,dtype=np.float64)\n    for f in range(10):\n        tr=fo!=f; te=fo==f\n        X=he[tr]; mu=X.mean(0); Xc=X-mu; Y=Zb[tr]; ym=Y.mean(0)\n        M=np.linalg.solve(Xc.T@Xc+lam*np.eye(768), Xc.T@(Y-ym)+lam*D.T).T; c=ym-M@mu\n        Lcv[te]=np.tanh(he[te]@M.T+c)@O.T+bo\n    pe=Lcv.argmax(1)\n    print('lam',lam,'seed',seed,'agree %.3f low %.3f acc %.3f'%((pe==pb).mean(),(pe==pb)[marg<np.quantile(marg,.2)].mean(),(pe==lab1).mean()))\n"
}
```


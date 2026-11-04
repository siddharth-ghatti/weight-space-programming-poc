## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nF={k:v.astype(np.float64) for k,v in np.load('feats_seq1024_depth.npz').items()}\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\nMd=np.concatenate([np.load('pass_40/mean_hidden_L8.npy'),np.load('pass_42/mean_hidden_L8.npy')]).astype(np.float64)\nCd=np.concatenate([np.load('pass_40/cls_hidden_L8.npy'),np.load('pass_42/cls_hidden_L8.npy')]).astype(np.float64)\nnp.savez('feats_drop6skip.npz', M8=Md, C8=Cd, M0=F['M0'])\nfor lam in [0.02, 0.03]:\n    for c in [10]:\n        r=[cv(Md,Cd,F['M0'],y,c=c,lam=lam,c0=0.5,seed=sd) for sd in range(5)]\n        print('drop6skip',c,lam,np.round(r,4),round(np.mean(r),4), flush=True)\nr=[cv(F['M9'],F['C9'],F['M0'],y,c=10,lam=0.02,c0=0.5,seed=sd) for sd in range(3,5)]\nprint('full seeds3,4',np.round(r,4))\n"
}
```


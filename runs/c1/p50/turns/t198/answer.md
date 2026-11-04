## reasoning summary

I'm going to screen the candidates on the subset first, comparing drop2+6, drop6, and full at the regularized settings. If drop2+6 falls well behind drop6, by something like 0.3 or more, I'll drop it without reading the rest.



## message

Next I'll compare drop-2+6 against drop-6 and the full model on the 1024 subset at the regularized settings. If it is clearly worse, I'll skip reading the remaining 472.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nd=np.load('feats_drop6skip.npz'); Md,Cd,M0=[d[k].astype(np.float64) for k in ['M8','C8','M0']]\nF={k:v.astype(np.float64) for k,v in np.load('feats_seq1024_depth.npz').items()}\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\ns=slice(0,1024); ys=y[s]\nM26=np.load('pass_43/mean_hidden_L7.npy').astype(np.float64); C26=np.load('pass_43/cls_hidden_L7.npy').astype(np.float64)\nprint('rel change vs drop6', np.linalg.norm((M26-Md[s])[:,KEEP])/np.linalg.norm((Md[s]-Md[s].mean(0))[:,KEEP]))\nfor c,lam,c0 in [(10,0.08,0.5),(5,0.05,0.25),(10,0.02,0.5)]:\n    r=[cv(M26,C26,M0[s],ys,c=c,lam=lam,c0=c0,seed=sd) for sd in range(6)]\n    r1=[cv(Md[s],Cd[s],M0[s],ys,c=c,lam=lam,c0=c0,seed=sd) for sd in range(6)]\n    print('subset',c,lam,c0,'drop2+6',round(np.mean(r),4),'drop6',round(np.mean(r1),4), flush=True)\n"
}
```


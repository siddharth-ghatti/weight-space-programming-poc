## reasoning summary

The full 1496-sample run at 10-fold with seed 0 came out at 98.26%. Next I want to sweep seeds and parameters, and get a base-feature baseline for comparison, which means checking whether the base M9, C9, and M0 features exist for all 1496 samples. I'll list the npz files to see what's already saved.



## message

The realized design scores 98.26% in 10-fold CV on the full 1496. Next I'll average over seeds, sweep c/lam a little, and find the base features for a same-protocol comparison.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nprint([p.name for p in WORK.iterdir() if p.suffix in ('.npz','.npy')])\nd=np.load('feats_seq1024.npz')\nids=[f'p{i:04d}' for i in range(1496)]\ny=labels_for(ids, practice, LABELS)\nM9,C9,M0=[d[k].astype(np.float64) for k in ['mean_hidden_L9','cls_hidden_L9','mean_hidden_L0']]\nfor c,lam,c0 in [(10,0.02,0.5),(10,0.01,0.5),(10,0.04,0.5),(10,0.02,1.0),(10,0.02,0.25),(20,0.02,0.5),(5,0.02,0.5)]:\n    r=[cv(M9,C9,M0,y,c=c,lam=lam,c0=c0,seed=s) for s in range(3)]\n    print(c,lam,c0, np.round(r,4), round(np.mean(r),4))\n"
}
```


## reasoning summary

Comparing the drop6 and full variants at higher regularization, the gap narrows, and at lambda 0.08 with C=10 drop6 edges out full. The difference seems quite sensitive to the hyperparameters, and with only 1496 samples the CV noise is probably large enough to matter. Now I'm going through the remaining settings, starting with (10, 0.02, 0.5).

Now I'm comparing the remaining settings: drop6 comes out ahead at higher regularization, around 98.25 on average versus about 98.3 for full. The gap is small, so with this little data I'm going to check the top drop6 settings across more seeds before choosing.



With stronger regularization (lam 0.05–0.08), drop-6 reaches about 98.26–98.30 against full-model scores of 98.22–98.40, so the real gap is nearer 0.05–0.1%. I'm running 10 seeds on the top drop-6 settings to confirm it.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nd=np.load('feats_drop6skip.npz'); Md,Cd,M0=[d[k].astype(np.float64) for k in ['M8','C8','M0']]\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\nfor c,lam,c0 in [(10,0.08,0.5),(10,0.12,0.5),(10,0.08,0.25),(5,0.05,0.25),(7,0.08,0.35)]:\n    r=[cv(Md,Cd,M0,y,c=c,lam=lam,c0=c0,seed=sd) for sd in range(10)]\n    print(c,lam,c0,'drop6',round(np.mean(r),4), round(np.std(r),4), flush=True)\n"
}
```


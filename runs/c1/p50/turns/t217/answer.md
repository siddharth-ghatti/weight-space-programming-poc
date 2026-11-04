## reasoning summary

Comparing the two cross-validation setups, the 20-fold numbers are closer to the final training setup, and both land around 98.1–98.3%. Drop46 looks to be roughly 98.2% overall with a margin of about 0.4–0.5%. Now I'm considering whether a few cheap changes could widen that margin for drop46.

I'm weighing a few cheap ways to add margin for drop46, like pulling in the mean and CLS features from the last layer, though extra linear carries from earlier layers didn't seem to help. Kernel tuning on a smaller 10-fold grid is another option, since 20-fold would cost about three times as much. Before I commit to that, the real question is whether drop46 or drop6 is the safer choice, which depends on the failure risk under each.

I'm now weighing whether drop46 or drop6 is the better bet, since the margin difference is small and the failure risk is roughly a coin flip. I'm also wondering if the pool could be improved without extra carries, so I'll try a small tuning grid on the kernel parameters with 10-fold, using paired seeds to keep the comparison fair despite the noise.



The two CV schemes disagree on drop4+6's gap, with 20-fold (−0.04%) closer to the final setting, so it's probably around 98.2–98.3. I'll now try a small grid of pool hyperparameters to see whether it can recover some margin.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib7 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS)\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\nfor (c,lam,c0) in [(10,.05,.5),(10,.1,.5),(10,.05,1.0),(10,.05,.25),(20,.1,.5),(5,.03,.5),(10,.2,1.0)]:\n    r=[cv(M,C,M0,y,c=c,lam=lam,c0=c0,k=10,seed=s) for s in range(10,15)]\n    print(c,lam,c0,np.round(r,4),round(np.mean(r),4),flush=True)"
}
```


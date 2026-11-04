## reasoning summary

I'm checking the drop46 full-1496 cross-validation, which is coming in around 98.1–98.2 across four seeds. The notes report higher numbers for drop6skip over ten seeds, so I want to run both on the same seeds to compare fairly, and also try c=20 and lam=0.08.



Drop4+6 scores 98.1–98.2 on the full set, but my earlier drop6skip figures (98.34–98.42) used different seeds and settings. I'm rerunning both feature sets on identical seeds and settings, c=20 and lam=0.08.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib7 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS)\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\nd=np.load('feats_drop6skip.npz'); Ma,Ca=d['M8'].astype(np.float64),d['C8'].astype(np.float64)\nres={}\nfor (c,lam,c0) in [(10,.08,.5),(20,.05,.5)]:\n    for nm,(MM,CC) in [('drop6skip',(Ma,Ca)),('drop46',(M,C))]:\n        r=[cv(MM,CC,M0,y,c=c,lam=lam,c0=c0,seed=s) for s in range(4)]\n        print(nm,c,lam,c0,np.round(r,4),round(np.mean(r),4))"
}
```


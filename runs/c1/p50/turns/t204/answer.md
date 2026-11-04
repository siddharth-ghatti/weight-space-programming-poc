## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib7 import *\nids=json.load(open('pass_44/meta.json'))['ids']\ny=labels_for(ids, practice, LABELS)\nd=np.load('feats_drop6skip.npz')\nallids=[r['id'] for r in practice()]\nassert allids[:1024]==ids\nM0=d['M0'][:1024]\nM_a=d['M8'][:1024]; C_a=d['C8'][:1024]\nM_b=np.load('pass_44/mean_hidden_L7.npy'); C_b=np.load('pass_44/cls_hidden_L7.npy')\nfor name,(M,C) in [('drop6skip',(M_a,C_a)),('drop46',(M_b,C_b))]:\n    for (c,lam,c0) in [(10,.02,.5),(5,.05,.25)]:\n        r=[cv(M,C,M0,y,c=c,lam=lam,c0=c0,seed=s) for s in range(5)]\n        print(name,c,lam,c0,np.round(r,4),np.mean(r))"
}
```


## reasoning summary

I need to insert WORK into sys.path.



## tool_use: run_python

```json
{
 "code": "import sys, time; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nd=np.load('feats_seq1024.npz')\nids=[f'p{i:04d}' for i in range(1496)]\nP=practice(); print(P[0]['id'], P[1495]['id'], len(P))\ny=labels_for(ids, practice, LABELS)\nM9,C9,M0=[d[k].astype(np.float64) for k in ['mean_hidden_L9','cls_hidden_L9','mean_hidden_L0']]\nt=time.time()\nprint(cv(M9,C9,M0,y,c=10,lam=0.02,c0=0.5), time.time()-t)\n"
}
```


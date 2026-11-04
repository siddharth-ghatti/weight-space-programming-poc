## reasoning summary

I'm checking the base kernel CV for M10 with the linear M0 term, comparing it against M9 and the pruned version, which is currently around 98.37.

I'm now going to measure how much the sentence-mean shifts at each layer, both in the base model and in the edited one on the kept dimensions, and check how similar the kernels are between consecutive layers. Layers that barely change look like the best candidates.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nb=np.load('base_feats.npz'); E=np.load('feats_seq1024_depth.npz')\ndef cka(X,Y):\n    X=X-X.mean(0); Y=Y-Y.mean(0)\n    return np.linalg.norm(X.T@Y)**2/(np.linalg.norm(X.T@X)*np.linalg.norm(Y.T@Y))\ndef rel(X,Y):\n    return np.linalg.norm(Y-X)/np.linalg.norm(X)\nfor l in range(12):\n    X=b[f'mean_hidden_L{l}'][:,KEEP].astype(np.float64); Y=b[f'mean_hidden_L{l+1}'][:,KEEP].astype(np.float64)\n    Xc=X-X.mean(0); Yc=Y-Y.mean(0)\n    print('base layer',l,'rel',round(rel(X,Y),3),'centered rel',round(rel(Xc,Yc),3),'cka',round(cka(X,Y),3))\nfor l in range(4,9):\n    X=E[f'M{l}'][:,KEEP].astype(np.float64); Y=E[f'M{l+1}'][:,KEEP].astype(np.float64)\n    Xc=X-X.mean(0); Yc=Y-Y.mean(0)\n    print('edited layer',l,'rel',round(rel(X,Y),3),'centered rel',round(rel(Xc,Yc),3),'cka',round(cka(X,Y),3))\n"
}
```


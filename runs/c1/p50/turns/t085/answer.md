## reasoning summary

I'm checking whether the base features cover all 1496 items, and whether the pass_19 set spans ids p0000 through p1023. If it does, I'll compare the base and edited model features on that first 1024-id subset, keeping the KEEP dimensions, and run a 10-fold kernel CV on both.



## message

Compare pruned (pass_19) vs base features on the same 1024 ids, layer by layer, using the honest RBF-cosine kernel CV on KEEP dims.

## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz')\nids19=json.load(open('pass_19/meta.json'))['ids']; P=practice(); id2i={p['id']:i for i,p in enumerate(P)}\nidx=np.array([id2i[i] for i in ids19]); yy=y[idx]\ndef cosK(X,g=1.0):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); X/=np.linalg.norm(X,axis=1,keepdims=True)\n    return np.exp(g*(X@X.T-1))\nfor l in [0,2,4,6,8,9]:\n    Xb=d[f'mean_hidden_L{l}'][idx]; Xe=np.load(f'pass_19/mean_hidden_L{l}.npy')\n    rel=np.linalg.norm(Xe-Xb)/np.linalg.norm(Xb-Xb.mean(0))\n    cs=np.mean(np.sum((Xe[:,KEEP]-Xe[:,KEEP].mean(0))*(Xb[:,KEEP]-Xb[:,KEEP].mean(0)),1)/np.linalg.norm(Xe[:,KEEP]-Xe[:,KEEP].mean(0),axis=1)/np.linalg.norm(Xb[:,KEEP]-Xb[:,KEEP].mean(0),axis=1))\n    rb=krr_cv(cosK(Xb),yy,[0.01],k=10,seeds=(0,)); re=krr_cv(cosK(Xe),yy,[0.01],k=10,seeds=(0,))\n    print(l, 'relerr %.3f centcos %.3f'%(rel,cs), 'base',rb,'edit',re)"
}
```


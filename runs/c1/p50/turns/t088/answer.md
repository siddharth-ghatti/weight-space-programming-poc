## reasoning summary

Working out how many layers I'd need: the small-k setup tops out near 97.9%, which isn't enough, and the jump at M9 looks real, so I need roughly nine layers of genuine computation. I'm also considering lowering g to 0.5, since my notes say the 0.5 to 1 range works best.

I'm checking whether dropping a middle layer could work instead, since BERT tolerates that fairly well, and I'm looking at the earlier pass_8/9 run that already dropped layers 4 and 6. Next I want to see what those pass_9 features gave, by comparing the kernel against the base model.



Small-k accuracy tops out near 97.9%, which isn't safe, so roughly nine layers of real BERT computation are needed. I'm testing whether dropping layers 4 and 6 holds up against the base model next.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz'); P=practice(); id2i={p['id']:i for i,p in enumerate(P)}\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef K(Xs,g=1.0):\n    Z=np.concatenate([cn(X) for X in Xs],1)/np.sqrt(len(Xs)); return np.exp(g*(Z@Z.T-1))\nfor pk in [9,5]:\n    m=json.load(open(f'pass_{pk}/meta.json')); idx=np.array([id2i[i] for i in m['ids']]); yy=y[idx]\n    print('pass',pk,m['layer_map'], m['structure'])\n    import glob\n    for f in sorted(glob.glob(f'pass_{pk}/mean_hidden_L*.npy'), key=lambda s:int(s.split('L')[-1][:-4])):\n        l=int(f.split('L')[-1][:-4]); X=np.load(f)\n        orig = m['layer_map'].get(str(l)) if l>0 else -1\n        if orig is None: continue\n        Xb=d[f'mean_hidden_L{orig+1}'][idx]\n        print(l,'orig',orig,'edit',krr_cv(K([X]),yy,[0.01],k=10,seeds=(0,)),'base',krr_cv(K([Xb]),yy,[0.01],k=10,seeds=(0,)), 'M0+edit',krr_cv(K([d['mean_hidden_L0'][idx],X]),yy,[0.01],k=10,seeds=(0,)))"
}
```


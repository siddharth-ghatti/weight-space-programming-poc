## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nF={k:v.astype(np.float64) for k,v in np.load('feats_seq1024_depth.npz').items()}\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\ns=slice(0,1024); ys=y[s]\ncands={'full':(F['M9'][s],F['C9'][s]),\n       'drop6':(np.load('pass_39/mean_hidden_L8.npy').astype(np.float64),np.load('pass_39/cls_hidden_L8.npy').astype(np.float64)),\n       'drop6skip':(np.load('pass_40/mean_hidden_L8.npy').astype(np.float64),np.load('pass_40/cls_hidden_L8.npy').astype(np.float64))}\n# sanity: L6 (output of orig 5) vs base M7?\nb=np.load('base_feats.npz')\nM6n=np.load('pass_40/mean_hidden_L6.npy').astype(np.float64)\nfor t in [6,7]:\n    Bt=b[f'mean_hidden_L{t}'][:1024].astype(np.float64)\n    print('new L5-out mean vs base M',t, np.linalg.norm((M6n-Bt)[:,KEEP])/np.linalg.norm((Bt-Bt.mean(0))[:,KEEP]))\nprint('old L6 (edited M7) vs base M7', np.linalg.norm((F['M7'][s]-b['mean_hidden_L7'][:1024])[:,KEEP])/np.linalg.norm((b['mean_hidden_L7'][:1024]-b['mean_hidden_L7'][:1024].mean(0))[:,KEEP]))\nfor nm,(MM,CC) in cands.items():\n    r=[cv(MM,CC,F['M0'][s],ys,c=10,lam=0.02,c0=0.5,seed=sd) for sd in range(5)]\n    print(nm,np.round(r,4),round(np.mean(r),4), flush=True)\n"
}
```


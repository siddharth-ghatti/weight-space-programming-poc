## tool_use: run_python

```json
{
 "code": "import json, os\nos.makedirs('feats', exist_ok=True)\nP = practice()\nlab = np.array([LABELS.index(p['label']) for p in P]); np.save('feats/y.npy', lab)\nfor kind in ['cls','mean']:\n    for l in range(13):\n        a = np.load(f'pass_2/{kind}_hidden_L{l}.npy'); b = np.load(f'pass_3/{kind}_hidden_L{l}.npy')\n        np.save(f'feats/{kind}_L{l}.npy', np.concatenate([a,b]))\nm2=json.load(open('pass_2/meta.json')); m3=json.load(open('pass_3/meta.json'))\nassert m2['texts']+m3['texts']==[p['text'] for p in P]\nrng=np.random.default_rng(0)\nfolds = np.zeros(len(P),int)\nfor c in range(8):\n    idx=np.where(lab==c)[0]; rng.shuffle(idx); folds[idx]=np.arange(len(idx))%5\nnp.save('feats/folds.npy',folds)\ndef ridge_cv(X,y,lams):\n    res={}\n    for lam in lams:\n        pred=np.zeros(len(y),int)\n        for f in range(5):\n            tr=folds!=f; te=~tr\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n            Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n            Y=np.eye(8)[y[tr]]-1/8\n            Wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]), Xt.T@Y)\n            pred[te]=(Xe@Wt).argmax(1)\n        res[lam]=(pred==y).mean()\n    return res\nlams=[1,10,100,1000]\nfor kind in ['cls','mean']:\n    for l in range(13):\n        X=np.load(f'feats/{kind}_L{l}.npy')\n        r=ridge_cv(X,lab,lams)\n        print(kind,l,{k:round(v,3) for k,v in r.items()})\n"
}
```


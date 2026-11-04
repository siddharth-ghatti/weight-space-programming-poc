## reasoning summary

Next I'm merging everything into one npz file and running ridge cross-validation for each layer. Reserved dimensions are zero at the layer 1 output, then layer 2 writes carried scores into them, so I'll exclude them from every layer.

I'm settling on a stratified 5-fold ridge classifier with centered one-hot targets, sweeping a few lambdas. I'm also considering whether to standardize features, since the original used penalty 300 on raw ones. With n=1496 and d=768, I'll solve it in dual form via SVD, and I'm saving the helper as `lib.py` in the work folder so later runs can import it.



I'm merging the reads into one file and writing a reusable ridge-CV helper, lib.py, next. After that I'll run 5-fold CV per layer for mean-pooled and [CLS] features, excluding the 8 reserved dims since they leak labels.



## tool_use: run_python

```json
{
 "code": "import json, numpy as np\nlib = r'''\nimport numpy as np, json\nLBL = ['coverage_grant','exclusion','premium_payment','deductible_limits','claims_procedure','definitions','term_cancellation_renewal','conditions_duties']\nRES = np.array([83,88,222,254,444,495,506,601])\nKEEP = np.setdiff1d(np.arange(768), RES)\ndef labels():\n    P=[json.loads(l) for l in open('practice.jsonl')]\n    return np.array([LBL.index(p['label']) for p in P])\ndef folds(y, k=5, seed=0):\n    rng=np.random.RandomState(seed); f=np.zeros(len(y),int)\n    for c in np.unique(y):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f[idx]=np.arange(len(idx))%k\n    return f\ndef ridge_fit(X, y, lam, C=8):\n    mu=X.mean(0); Xc=X-mu; Y=np.eye(C)[y]; Y=Y-Y.mean(0)\n    d=X.shape[1]\n    W=np.linalg.solve(Xc.T@Xc+lam*np.eye(d), Xc.T@Y)\n    b=-mu@W + np.eye(C)[y].mean(0)\n    return W,b\ndef cv_acc(X, y, lams=(1,10,100,1000), k=5, seeds=(0,), std=False, ret_scores=False):\n    out={}\n    if std: X=(X-X.mean(0))/(X.std(0)+1e-6)\n    for lam in lams:\n        accs=[]; S=np.zeros((len(y),8))\n        for s in seeds:\n            f=folds(y,k,s)\n            for i in range(k):\n                tr=f!=i; te=f==i\n                W,b=ridge_fit(X[tr],y[tr],lam); S[te]=X[te]@W+b\n            accs.append((S.argmax(1)==y).mean())\n        out[lam]=float(np.mean(accs))\n        if ret_scores: out[('S',lam)]=S.copy()\n    return out\ndef load_base():\n    d=np.load('base_feats.npz'); return d\n'''\nopen('lib.py','w').write(lib)\nD={}\nfor kind in ['mean_hidden','cls_hidden']:\n    for l in range(13):\n        D[f'{kind}_L{l}']=np.concatenate([np.load(f'pass_2/{kind}_L{l}.npy'),np.load(f'pass_3/{kind}_L{l}.npy')])\nD['pooled']=np.concatenate([np.load('pass_2/pooled.npy'),np.load('pass_3/pooled.npy')])\nD['logits']=np.concatenate([np.load('pass_2/logits.npy'),np.load('pass_3/logits.npy')])\nm2=json.load(open('pass_2/meta.json')); print(m2.keys()); \nnp.savez('base_feats.npz',**D)\nimport sys; sys.path.insert(0,'.'); from lib import *\ny=labels(); print('base practice acc', (D['logits'].argmax(1)==y).mean())\nprint('reserved dims at L2 mean', np.abs(D['mean_hidden_L2'][:,RES]).max(), 'L3', np.abs(D['mean_hidden_L3'][:,RES]).max())"
}
```


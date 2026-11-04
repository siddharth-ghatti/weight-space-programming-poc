## reasoning summary

I'm checking that the second and third passes line up with the earlier pass_1 run, and that looks right. Next I'll merge everything into feats.npz and score a cross-validated ridge classifier on each layer, for both mean and cls features. The plan is stratified 5-fold, standardized features, one-hot ridge regression in closed form, and an LDA with shrinkage alongside it.



Both reads are done, and I'm combining them into one array file to score each layer's mean and CLS features with a cross-validated ridge classifier next.



## tool_use: run_python

```json
{
 "code": "import json\nP=practice(); lab={l:i for i,l in enumerate(LABELS)}\ny=np.array([lab[p['label']] for p in P])\nM={}; C={}\nfor L in range(13):\n    M[L]=np.concatenate([np.load(f'pass_2/mean_hidden_L{L}.npy'),np.load(f'pass_3/mean_hidden_L{L}.npy')])\n    C[L]=np.concatenate([np.load(f'pass_2/cls_hidden_L{L}.npy'),np.load(f'pass_3/cls_hidden_L{L}.npy')])\nmeta2=json.load(open('pass_2/meta.json')); meta3=json.load(open('pass_3/meta.json'))\ntexts=meta2['texts']+meta3['texts']\nassert texts==[p['text'] for p in P]\nnp.savez('feats.npz', y=y, **{f'M{L}':M[L] for L in range(13)}, **{f'C{L}':C[L] for L in range(13)})\nrng=np.random.RandomState(0); folds=np.zeros(len(y),int)\nfor c in range(8):\n    idx=np.where(y==c)[0]; rng.shuffle(idx); folds[idx]=np.arange(len(idx))%5\nnp.save('folds.npy',folds)\ndef ridge_cv(X,y,lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n        Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd\n        Y=np.eye(8)[y[tr]]-1/8\n        A=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(X.shape[1]), Xtr.T@Y)\n        accs.append((np.argmax(Xte@A,1)==y[te]).mean())\n    return np.mean(accs)\nfor L in range(13):\n    print(L, 'mean', [round(ridge_cv(M[L],y,l),3) for l in [10,100,1000]], 'cls', [round(ridge_cv(C[L],y,l),3) for l in [10,100,1000]])"
}
```


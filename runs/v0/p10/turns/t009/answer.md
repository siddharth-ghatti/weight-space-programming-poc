## reasoning summary

I'm seeing layers 8 to 11 perform about the same, around 0.95 to 0.96, so the gap is small. Combining layer 0 with them might push it to roughly 0.97. I want to test the pairs (0,8) through (0,11) more carefully, and check whether summing logits behaves like ridge on the concatenated features. I'm also wondering whether the layer 0 "bag of embedding" signal could be approximated by something available at the final layer.

I'm considering whether a uniform attention head at layer 0 could push a shared context vector into every token, so the final CLS picks up the mean embedding signal. The shift would be the same for every token, but how much it disrupts the later layers is unclear. I'm also weighing a positional route: projecting out the CLS direction from one head's query so CLS attends nearly uniformly.

I'm weighing whether to quantify the layer combinations first, to see if the extra complexity of the layer-0 mean approach is worth it. The sum-based combination might lose the concatenated structure, so I'll test that before going further.



Layers 8–10 mean-pooled reach about 96%, and L11/L12 about 95.4%. I'm next checking whether the multi-layer combinations add anything under repeated cross-validation, and whether mixing CLS and mean pooling helps.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in range(13)}\nCc = {l: np.load(f'pass_2/cls_hidden_L{l}.npy').astype(np.float64) for l in [11,12]}\ndef cv_acc(X, y, lam, reps=6):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nfor combo in [(0,8),(0,9),(0,11),(1,9),(2,9),(3,9),(4,9),(0,4,9)]:\n    X = np.concatenate([M[c] for c in combo],1)\n    print(combo, [cv_acc(X,y,lam) for lam in [200,400,800]])\nX = np.concatenate([Cc[11], M[11]],1); print('cls11+mean11', [cv_acc(X,y,lam) for lam in [200,400,800]])\nX = Cc[11]+M[11]*3; print('cls11+3mean11 sum', [cv_acc(X,y,lam) for lam in [100,200,400]])\n"
}
```


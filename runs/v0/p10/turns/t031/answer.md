## reasoning summary

I'm comparing the pooled-token combinations: cls9 with mean9 scores 0.9663, a 0.7% gain over mean9 alone at 0.959 on the base 1024 data, and cls10 with mean8 and cls9 with mean8 come in close behind. I'm also sketching a design where layers 0 through 8 stay untouched, so every token, including CLS, carries a normal h9, and only the pooling step at layer index 9 changes.

I'm working out a residual design where CLS keeps its own state through a projection onto a subspace S1, while mean-pooled information goes into S1 and CLS's own state stays in the orthogonal complement S2. I'm planning to simulate this offline on base data, with a centering bias option and a layer-norm step, and to try different scaling values and subspace sizes.

I'm now weighing a second design where attention heads split the work: some uniformly pool the mean into a 384-dim subspace, others attend only to CLS and project onto its top PCA directions. I'm noting that the earlier PCA truncation test lost about 0.2–0.5%, so I'll simulate both this routing approach and the residual one to compare how much information each keeps.



Concatenating the [CLS] token with mean pooling at layer 9 scores 96.6%, versus 95.9% for mean pooling alone. I'll simulate a residual-mixing variant offline to pick the scale and subspace size, keeping layers 0–8 unchanged.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nm = np.load('pass_2/mean_hidden_L9.npy').astype(np.float64); x = np.load('pass_2/cls_hidden_L9.npy').astype(np.float64)\nLN = lambda X: (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)\ndef cv_acc(X, y, lam, reps=4):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nmbar = m.mean(0); U,S,Vt = np.linalg.svd(m-mbar, full_matrices=False)\nfor k in [384, 512, 640]:\n    P1 = Vt[:k].T\n    for c in [1,3,10]:\n        f = LN(x + c*((m-mbar)@P1)@P1.T)\n        print(k, c, [cv_acc(f,y,lam) for lam in [100,200,400]])\n"
}
```


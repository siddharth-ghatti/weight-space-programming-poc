## reasoning summary

Mean pooling across layers 8 to 12 reaches about 96%, and even layer 0 alone, a plain bag of embeddings, gets 95%, while CLS at layer 12 sits near 92.8%. Next I want to tune lambda more finely, test whether concatenating layers helps, and try LDA as well.

I'm checking whether the mean-of-embeddings idea needs its own computation, but I'll avoid that since it's the embedding layer rather than the forward pass. Next I'm trying a finer lambda grid from 30 up to 500 on layers 8 to 12, plus concatenations of mean layer 8 with layer 12, and layer 0 with layer 12, to see if combining layers helps.



Mean-pooling beats the CLS token clearly (about 96% for layers 8–12 against 92.8% for CLS at L12). I'm next tuning the regularisation more finely and checking whether concatenating layers is worth the extra complexity.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nrng = np.random.default_rng(0); perm = rng.permutation(1024); folds = np.array_split(perm, 5)\ndef ridge_cv(X, y, lams, std=True):\n    res = {}\n    n,d = X.shape\n    for lam in lams:\n        correct = 0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else 1\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            correct += ((Xv@Wt).argmax(1)==y[f]).sum()\n        res[lam] = round(correct/n,3)\n    return res\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in range(13)}\nlams=[30,60,100,200,400]\nfor l in [5,6,8,9,10,11,12]:\n    print('mean',l, ridge_cv(M[l],y,lams), 'nostd', ridge_cv(M[l],y,[1,3,10,30],std=False))\nfor combo in [(0,12),(8,12),(4,8,12),(0,6,12),(9,12),(0,9)]:\n    X = np.concatenate([M[c] for c in combo],1)\n    print(combo, ridge_cv(X,y,[100,300,1000]))\n"
}
```


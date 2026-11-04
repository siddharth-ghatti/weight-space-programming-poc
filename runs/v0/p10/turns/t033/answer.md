## reasoning summary

The LN'd routed PCA features are doing poorly at 0.93–0.94. Layer norm rescales each sample by its own std, and since the PCA features are centered, that scale varies a lot from sample to sample and distorts them. The nostd lambda values are also off now, because after LN each dimension has small magnitude, so lam=1..10 is too strong and causes underfitting. I'll drop the LN on the centered features.

I'm weighing whether to keep the features uncentered, so the mean term stabilizes the per-sample norm. I'm also considering a constant bias dimension to keep the LN scale steady, and I'll need to check whether standardization handles the per-dimension scaling.

I'm checking whether dropping the centering would cause a problem. With a large constant offset in the spare dimensions, the layer-norm scale would stay nearly fixed, so the normalization would act roughly linearly. Then I'd just compare the plain concatenated PCA features with and without the layer norm, using both standardized and unstandardized versions.



The LayerNorm stand-in is distorting the results because the routed features are centred and their normalisation scale swings; I'll re-simulate with a bias that keeps LayerNorm near-linear to isolate the PCA truncation cost.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nm = np.load('pass_2/mean_hidden_L9.npy').astype(np.float64); x = np.load('pass_2/cls_hidden_L9.npy').astype(np.float64)\nLN = lambda X: (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)\ndef cv_acc(X, y, lam, reps=4, std=True):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else np.ones(d)\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\ndef pcs(X): \n    Xb = X.mean(0); U,S,Vt = np.linalg.svd(X-Xb, full_matrices=False); return Xb, Vt\nmb, Vm = pcs(m); xb, Vx = pcs(x)\nfor k1,k2 in [(380,380),(500,260),(560,200)]:\n    F = np.concatenate([(m-mb)@Vm[:k1].T, (x-xb)@Vx[:k2].T],1)\n    print(k1,k2,'plain std',[cv_acc(F,y,l) for l in [100,200,400]], 'nostd', [cv_acc(F,y,l,std=False) for l in [3,10,30]])\n# in-space projection (no PCA truncation): mean in full 768 coords is impossible with cls too; check k1=768-k2 with cls small\n"
}
```


## reasoning summary

I'm finding that combining the layer-11 CLS token with mean-pooled layer-8 features gives 0.967, a bit better than the 0.963 from layer-8 mean pooling alone. The layer-11 CLS state is the usual one for the base model. I'm wondering whether I can restructure layer 11 so it produces that concatenation directly, but mean-pooling layer 8 seems to be what matters, since pooling layer 11 instead only reaches 0.9577.

I'm weighing a design where layer 8 pools the non-CLS tokens uniformly so that layer 11 can see the mean of h8 while the CLS token still runs normally. The difficulty is that a linear query can't zero out the non-CLS tokens and keep CLS intact at the same time.

I'm working out a rank-1 projection on the query matrix that would send non-CLS tokens to a uniform pool while leaving CLS close to its normal query, with the bias term folded in so the CLS query stays roughly intact.

I'm trying the idea of getting CLS to keep its normal attention while every other token pools uniformly at layer 8, using a rank-1 query projection. That would leave CLS carrying its own state plus a mean of layer 8, but the shared value and output maps make it unclear whether CLS can still look normal, so I'm weighing whether separate subspaces could hold the two pieces of information.

I'm testing whether cls9 or cls8 combined with mean8 holds up, since the CLS state entering layer 8 is untouched by pooling. If cls8 + mean8 works, one design would be to have layer 8's first six heads pool into a mean(h8) subspace while the other six route CLS into a second subspace.

I'm checking whether the CLS token's signal at layer 8 is separable enough to route it through a second group of heads, by measuring the margin between the CLS detector score and the largest token score. I also want to compare cross-validated accuracy for concatenating PCA-reduced mean8 with PCA-reduced cls8 against the other options.



Concatenating the normal [CLS] state with mean(h8) reaches about 96.7% versus 96.3%, so I'll next test cls8/cls9 plus mean8, including with each block cut to 384 dimensions.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM8 = np.load('pass_2/mean_hidden_L8.npy').astype(np.float64)\nCl = {l: np.load(f'pass_2/cls_hidden_L{l}.npy').astype(np.float64) for l in [6,7,8,9]}\ndef cv_acc(X, y, lam, reps=4, std=True):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else np.ones(d)\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nfor l in [6,7,8,9]:\n    print(f'cls{l}+mean8', [cv_acc(np.concatenate([Cl[l],M8],1),y,lam) for lam in [200,400,800]])\ndef pca(X,k):\n    Xc = X-X.mean(0); U,S,Vt = np.linalg.svd(Xc,full_matrices=False); return Xc@Vt[:k].T\nfor k1,k2 in [(384,384),(512,256)]:\n    Z = np.concatenate([pca(M8,k1), pca(Cl[8],k2)],1)\n    print('pca',k1,k2, [cv_acc(Z,y,lam) for lam in [200,400,800]], 'nostd', [cv_acc(Z,y,lam,std=False) for lam in [3,10,30]])\n"
}
```


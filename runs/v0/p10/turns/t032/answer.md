## reasoning summary

Residual mixing comes out around 0.96, no better than mean9 alone at 0.959, so the benefit from concatenation disappears once the parts are mixed. Mixing looks like a dead end, so I'll test clean routing instead, with the two parts kept in separate subspaces.

Routing the two pooled parts into separate subspaces, with the first part in dimensions 0 to k1 and the second in k1 to k1+k2, looks like the way forward. Earlier, standardized PCA features did poorly at 0.94 while unstandardized ridge reached 0.962, so I'll run cls9+mean9 at k1=k2=384 both ways, plus some whitening variants.



Residual mixing gives no gain (still about 96.0%); the [CLS] signal needs its own clean subspace, so I'm simulating routing with 6 heads for the mean and 6 for [CLS] under both ridge variants.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nm = np.load('pass_2/mean_hidden_L9.npy').astype(np.float64); x = np.load('pass_2/cls_hidden_L9.npy').astype(np.float64)\nLN = lambda X: (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)\ndef cv_acc(X, y, lam, reps=4, std=True):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else np.ones(d)\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\ndef pcs(X): \n    Xb = X.mean(0); U,S,Vt = np.linalg.svd(X-Xb, full_matrices=False); return Xb, Vt\nmb, Vm = pcs(m); xb, Vx = pcs(x)\nprint('ref mean9 nostd', [cv_acc(m,y,l,std=False) for l in [3,10,30]])\nprint('ref concat full nostd', [cv_acc(np.concatenate([m,x],1),y,l,std=False) for l in [3,10,30]])\nfor k1,k2 in [(384,384),(512,256)]:\n    for c2 in [1,2]:\n        F = np.concatenate([(m-mb)@Vm[:k1].T, c2*(x-xb)@Vx[:k2].T],1)\n        F = LN(F)  # stand-in for LayerNorm of the routed vector\n        print(k1,k2,c2,'std',[cv_acc(F,y,l) for l in [200,400]], 'nostd', [cv_acc(F,y,l,std=False) for l in [1,3,10]])\n"
}
```


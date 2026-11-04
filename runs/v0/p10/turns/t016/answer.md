## reasoning summary

The gain seems to need a lot of the first-layer information, around 256 or more dimensions, since an 8-dimensional injection can't carry it. That points away from a simple injection and toward a different feature set combined with ridge regularization, where many low-variance dimensions matter. A small injected perturbation would likely get swamped by those.

Now I'm weighing whether a dedicated CLS carrier at full strength would damage the middle layers, and whether making the CLS key stay stable through layers 1 to 4 could prevent that.

I'm now checking whether pooling the middle-layer mean into the CLS token at layer 8 and then pooling again at layer 11 could recover the late-layer information, since the CLS token is rarely attended from layer 5 onward.

Next I'm going to tabulate the pairs (a, 11) for each earlier layer a from 4 to 10, since pooling h11 at the final layer is available and the CLS could carry mean(h_a) forward.

The idea now is to have the CLS token keep running as a pooled summary rather than being overwritten. If CLS attends only to itself after injection, its state just evolves through LayerNorm and the FFN, which changes the info but keeps it. If it attends uniformly across layers instead, it would accumulate a mix of mean-pooled states from each layer together with its own state.

Simplest version first: a single pooling of layer a into the CLS, rather than a multi-layer running mix. I'll tabulate pairs (a, 11) for a from 4 to 10 before deciding whether the extra complexity is worth it.



The layer-0 gain needs more dimensions than a small injected vector can carry, so I'm dropping the bag-of-words injection. Next I'll test pooling two deep layers into [CLS], since attention to [CLS] is low after layer 4.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in range(13)}\ndef cv_acc(X, y, lam, reps=4):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nfor a,b in [(4,11),(5,11),(6,11),(8,11),(5,10),(4,9),(5,9),(6,9)]:\n    X = np.concatenate([M[a],M[b]],1)\n    print((a,b), [cv_acc(X,y,lam) for lam in [200,400,800]])\n"
}
```


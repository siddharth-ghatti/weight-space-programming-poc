## reasoning summary

Mean pooling already looks strong, at around 95-96% for layers 8-12 with lam=10, and even layer 0 reaches 93.8%. I'm considering whether concatenating mean L10 with L0 helps, along with a finer lambda grid. One design option on the table is the mean of the layer 11 output.

I'm weighing a few ways to get a mean-pooled signal from the final layers, and I'm leaning toward the option where the mean of layer 11's output feeds into layer 12's uniform attention. The option of taking the mean of hidden index 12 doesn't work, since that's the final output and isn't directly accessible at CLS. I'm going to test a few combinations quickly to see whether any of them beat the single-layer results.



Mean pooling outperforms [CLS] at about 95–96% CV accuracy on layers 8–12, with even layer 0 reaching roughly 94%. I'm next testing whether concatenating layers or a finer ridge grid helps.



## tool_use: run_python

```json
{
 "code": "y = lab[:1024]\ndef load(kind, l, d='pass_2'): return np.load(f'{d}/{kind}_hidden_L{l}.npy').astype(np.float64)\nrng = np.random.RandomState(0); perm = rng.permutation(1024); folds = np.array_split(perm,5)\ndef ridge_cv(X, y, lams):\n    res = {}\n    for lam in lams:\n        correct=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(len(y)), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv=(X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            A = Xt.T@Xt + lam*len(tr)*np.eye(X.shape[1])\n            Wt = np.linalg.solve(A, Xt.T@(Y-Y.mean(0)))\n            pred = (Xv@Wt).argmax(1)\n            correct += (pred==y[f]).sum()\n        res[lam]=round(correct/len(y),3)\n    return res\nlams=(0.03,0.1,0.3,1)\nfor l in [8,9,10,11]:\n    print(l, ridge_cv(load('mean',l), y, lams))\nfor combo in [(0,11),(0,10),(6,11),(8,11),(4,8,11),(0,6,11),(10,11),(11,12)]:\n    X = np.concatenate([load('mean',l) for l in combo],1)\n    print(combo, ridge_cv(X,y,lams))\n"
}
```


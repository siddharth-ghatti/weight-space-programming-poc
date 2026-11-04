## reasoning summary

The 10-fold cross-validation came back at 97.2%, with the errors spread fairly evenly, though term life shows up disproportionately (13 of 42). I think the model is in good shape. For the remaining passes I'm considering adding handwritten sentences aimed at term life and the categories it gets confused with, though the payoff is uncertain and my phrasing may not match the label conventions.

I'm weighing a targeted experiment that combines a CLS-based classifier with a mean-pooled layer 8 representation. The idea is to keep CLS normal through layer 9 while carrying the mean of layer 8 forward, which could give a small gain over the baseline, though I still need to work out how to compute and pass the mean through the remaining layers.

I'm dropping the idea of a mixed CLS and mean-pooled design at layer 8: the non-CLS tokens would end up as nonlinear functions of a shared constant, so their mean wouldn't match mean9 cleanly, and the concatenation with mean9 likely wouldn't beat the 0.963 baseline. I'm setting the architecture aside.

I'm considering whether a larger gain could come from carrying bag-of-words information through the deep layers, perhaps via attention weights that are sharpened along several directions, since the softmax would add the nonlinearity that the baseline lacks.

I'm weighing whether the weak L0 signal comes from word presence, which mean pooling already captures linearly, or from something length-related, since averaging position embeddings would encode sentence length. To tell these apart, I want to test whether adding the mean of layer 0 to the layer 8 mean actually helps.

I'm checking whether the length information matters by testing a concat of mean8 with the sequence length T, its reciprocal, and log T.



Ten-fold CV accuracy is 97.2% with errors spread evenly across classes and registers. Next I'll test whether the layer-0 gain is just sentence length by adding mean8 and length features.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM8 = np.load('pass_2/mean_hidden_L8.npy').astype(np.float64)\nT = np.array([len(t['ids']) for t in tokenize([p['text'] for p in P[:1024]])], float)\ndef cv_acc(X, y, lam, reps=4):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nLf = np.stack([T,1/T,np.log(T)],1)\nprint('mean8', [cv_acc(M8,y,l) for l in [200,400]])\nprint('mean8+len', [cv_acc(np.concatenate([M8, np.repeat(Lf,10,1)],1),y,l) for l in [200,400]])\n"
}
```


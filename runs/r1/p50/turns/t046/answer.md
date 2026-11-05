## reasoning summary

I'm sorting the sequences by length and picking out the longest 32 and the next 32. Since 17 of the top 128 already overlap with the existing probe set, I'll exclude those and set up two new batches, about 9-10k and 6k tokens. That gives 10 passes across 5 layers, each layer getting roughly 21k tokens in total.

I'm weighing whether 4× the data is worth the cost, since the regression has many parameters and the gap in sample size is large. Since the compensation matrix is big and the kept units are fixed, I'll check how test error shrinks as the training fraction grows. I'll fit on 25%, 50%, and 75% of the sequences with a fixed 25% holdout, starting with L1 at m=1024 and lambda=1.



The 32 longest paragraphs hold about 10,268 tokens, roughly double the current probe set. Before spending more passes, I'll run a learning curve on L1 and L3 to see whether 3–4× more tokens is worth it.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('builder.py').read())\niscls = np.load('ffn_in_L0_iscls.npy'); seq = np.cumsum(iscls)-1\nte = seq%4==3\ndef curve(L, m, lam_rel, frac_seqs):\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\"); b2 = base(pre+\"output.dense.bias\")\n    A = ffn_data(L); H = gelu(A@W1.T+b1); Y = H@W2.T; Z = A+Y\n    res=[]\n    for fs in frac_seqs:\n        tr = np.isin(seq%4, fs)\n        sc = H[tr].var(0)*(W2**2).sum(0)\n        keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\n        T = H[:,drop]@W2[:,drop].T; Hk = H[:,keep]; mk = Hk[tr].mean(0); mt = T[tr].mean(0)\n        Xk = Hk[tr]-mk; G = Xk.T@Xk; lam = lam_rel*np.trace(G)/m\n        B = np.linalg.solve(G+lam*np.eye(m), Xk.T@(T[tr]-mt))\n        err = T - ((Hk-mk)@B+mt)\n        den = np.sqrt(((Z[te]-Z[te].mean(0))**2).sum(1)).mean()\n        res.append((tr.sum(), np.sqrt((err[te]**2).sum(1)).mean()/den))\n    return res\nfor L in [1,3]:\n    for lr in [1, 0.3]:\n        print(L, lr, [(n, round(e,4)) for n,e in curve(L,1024,lr,[[0],[0,1],[0,1,2]])])"
}
```


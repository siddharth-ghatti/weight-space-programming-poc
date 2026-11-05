## reasoning summary

I need to tie each token back to its sequence so I can hold out whole sequences, so I'm building a sequence index from the cumulative sum of the CLS markers. Then I'll evaluate layer L at m=1024 and m=1536, fitting variance-rank ridge with three lambda settings on 16 sequences and measuring relative reconstruction error on the other 16.

I'm also going to measure the error against the LN2 input, a + FFN(a), since that's the quantity that feeds the normalization, and compare the deviation in the normalized output.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('builder.py').read())\niscls = np.load('ffn_in_L0_iscls.npy'); seq = np.cumsum(iscls)-1\ntr = seq%2==0; te = ~tr\ndef evalL(L, m, lam_rel, score='var'):\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\"); b2 = base(pre+\"output.dense.bias\")\n    A = ffn_data(L); H = gelu(A@W1.T+b1)\n    Y = H@W2.T\n    sc = H[tr].var(0)*(W2**2).sum(0)\n    keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\n    T = H[:,drop]@W2[:,drop].T\n    Hk = H[:,keep]; mk = Hk[tr].mean(0); mt = T[tr].mean(0)\n    Xk = Hk[tr]-mk; G = Xk.T@Xk; lam = lam_rel*np.trace(G)/m\n    B = np.linalg.solve(G+lam*np.eye(m), Xk.T@(T[tr]-mt))\n    pred = (Hk-mk)@B + mt\n    err = T - pred\n    # relative to residual stream scale (a+Y varying)\n    Z = A+Y+b2\n    zs = (Z-Z.mean(1,keepdims=True)).std(1)\n    def r(mask): return np.sqrt((err[mask]**2).sum(1)).mean()/ np.sqrt(((Z[mask]-Z[mask].mean(0))**2).sum(1)).mean()\n    e0 = T - T[tr].mean(0)\n    def r0(mask): return np.sqrt((e0[mask]**2).sum(1)).mean()/ np.sqrt(((Z[mask]-Z[mask].mean(0))**2).sum(1)).mean()\n    return r(tr), r(te), r0(te)\nfor L in range(5):\n    for m in [1024,1536]:\n        out = [evalL(L,m,lr) for lr in [10,1,0.1]]\n        print(L, m, ' '.join(f\"lam{lr}: tr{a:.4f} te{b:.4f} (nocomp te {c:.4f})\" for lr,(a,b,c) in zip([10,1,0.1],out)))"
}
```


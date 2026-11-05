## reasoning summary

Pruning to 1024 units leaves about 20% relative residual error per layer, and that error compounds across the 6 layers. Ridge regression of the dropped contributions onto the kept activations might reduce it, so I'll test that by fitting the compensation matrix on training tokens and evaluating on test tokens. With 3

I'm weighing whether 800 training tokens against 1024 parameters per output will overfit, so I'll lean on ridge regularization. A selection that accounts for downstream impact would be better but hard to compute, so I'll keep the selection simple. A merge-based dimension reduction doesn't work exactly here because of the GELU nonlinearity, so I'm going to test least-squares compensation first.



FFN contributions are spread across many units, so keeping 1024 leaves about 20% error per layer that compounds over six layers. Next I'll test least-squares compensation: refitting the kept units' output weights to absorb the dropped units' predictable contribution, scored on a held-out quarter.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\nM = np.load(WORK/'pass_4/attention_mask.npy').astype(bool)\npara = np.repeat(np.arange(32)[:,None], M.shape[1], 1)[M]\ntr = para%4!=0; te = ~tr\nfor Lr in [1,3,5]:\n    pre = f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\"); b2 = base(pre+\"output.dense.bias\")\n    if Lr<5:\n        A = np.load(WORK/f'ffn_in_L{Lr}.npy'); trm, tem = tr, te\n    else:\n        A = np.load(WORK/'ffn_in_L5_cls.npy'); idx = np.arange(len(A)); trm = idx%4!=0; tem = ~trm\n    H = gelu(A@W1.T + b1); R = A + H@W2.T + b2; Rc = R - R.mean(1, keepdims=True)\n    var = H[trm].var(0)*(W2**2).sum(0)\n    line=[]\n    for m in [512,1024,1536]:\n        keep = np.argsort(-var)[:m]; drop = np.setdiff1d(np.arange(3072), keep)\n        Hk = H[:,keep]; mk = Hk[trm].mean(0); T = H[:,drop]@W2[:,drop].T; mt = T[trm].mean(0)\n        Xk = Hk[trm]-mk; Tt = T[trm]-mt\n        G = Xk.T@Xk\n        for lam_rel in [1e-3,1e-2,1e-1,1]:\n            lam = lam_rel*np.trace(G)/m\n            B = np.linalg.solve(G+lam*np.eye(m), Xk.T@Tt)\n            D = T[tem]-mt - (Hk[tem]-mk)@B\n            Dc = D - D.mean(1,keepdims=True)\n            rel = np.linalg.norm(Dc,axis=1)/np.linalg.norm(Rc[tem],axis=1)\n            line.append(f\"m{m} lam{lam_rel}:{rel.mean():.3f}\")\n    print(f\"L{Lr}: \"+' '.join(line))"
}
```


## reasoning summary

I'm setting up a holdout test for L0: the old probe set gets split in half, with new data plus half the old used for training and the other half for testing. I'll compare that against the earlier 5049-only setup, then pick lambda and fit at m=1024, and check how m=1024, 1280, and 1536 compare on error.



Both batches are in; I'm extracting the FFN inputs next and measuring L0 held-out error with the extra data to pick the ridge strength.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nextract_probe(0, 'pass_14', 1, 'B1'); extract_probe(0, 'pass_15', 1, 'B2')\nA, S = all_data(0); print(A.shape)\n# sanity: L0 FFN inputs are base inputs; compare any overlapping? none. holdout: old seqs odd\nte = (S<100) & (S%2==1)\ndef holdout(L, m, lam_rel, A, S, te, use_new=True):\n    tr = ~te if use_new else ((S<100)&~te)\n    keep,B,mk,mt = fit_layer(L, m, lam_rel, A=A[tr], save=False)\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64)\n    At = A[te].astype(np.float64); H = gelu(At@W1.T+b1); Y = H@W2.T; Z = At+Y\n    drop = np.setdiff1d(np.arange(3072), keep)\n    T = H[:,drop]@W2[:,drop].T; pred = (H[:,keep]-mk)@B + mt\n    den = np.sqrt(((Z-Z.mean(0))**2).sum(1)).mean()\n    return np.sqrt(((T-pred)**2).sum(1)).mean()/den, np.sqrt(((T-T.mean(0))**2).sum(1)).mean()/den\nfor m in [1024]:\n    print('old only lam1', holdout(0,m,1.0,A,S,te,False))\n    for lr in [1.0,0.3,0.1]:\n        print(m, lr, holdout(0,m,lr,A,S,te,True))"
}
```


## reasoning summary

Next I'll fit L4, then L5. For L5 only the CLS representation matters, so I'll re-probe it on the edited model and then fit at m=1024 with CV-chosen lambda, using both the edited and base CLS samples.

I'm weighing whether to match L5's FFN output directly or fit to logit error, since the head is nonlinear after LN2. I'll keep the simple FFN-output matching, and set up the L5 probe after fitting L4.



The layer-4 data is in, so I'm fitting layer 4 next. Then I'll set up the layer-5 probe, which needs only the `<s>` position, in a single pass over all 1000 ids.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nL=4\nextract_probe(L, 'pass_22', 5, 'B1'); extract_probe(L, 'pass_23', 5, 'B2')\nA, S = all_data(L); te = (S<100) & (S%2==1)\ndef holdout(L, m, lam_rel, A, S, te):\n    tr = ~te\n    keep,B,mk,mt = fit_layer(L, m, lam_rel, A=A[tr], save=False)\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64)\n    At = A[te].astype(np.float64); H = gelu(At@W1.T+b1); Y = H@W2.T; Z = At+Y\n    drop = np.setdiff1d(np.arange(3072), keep)\n    T = H[:,drop]@W2[:,drop].T; pred = (H[:,keep]-mk)@B + mt\n    den = np.sqrt(((Z-Z.mean(0))**2).sum(1)).mean()\n    return round(np.sqrt(((T-pred)**2).sum(1)).mean()/den,4), round(np.sqrt(((T-T.mean(0))**2).sum(1)).mean()/den,4)\nprint([(lr, holdout(L,1024,lr,A,S,te)) for lr in [0.1]])\nrestore_layer(L)\nfit_layer(L, 1024, 0.05)\nprint(setup_probe(L+1, 1024)); print(load_keeps().keys())\nids = json.load(open(WORK/'all_ids.json')); print(len(ids))"
}
```


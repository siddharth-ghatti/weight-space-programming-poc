## reasoning summary

I'm looking at the no-compression L1 model with m=1024 on the other split, where the test error is around 0.34. As the data grows (1229 to 3904 samples), the error drops only slowly, from 0.336 to 0.307. Extrapolating toward 20k samples gives maybe 0.28, and the training error suggests a floor near 0.25 to 0.26, so the gain from quadrupling the data is roughly 20%. I'm also starting to check how the error denominators compare.

I'm weighing whether the selection method might matter more than the data size, so I'm sketching an OBS-style greedy backward elimination with refitting, fit on the training split and checked on the holdout. I'm working out the update rule for the weights and the inverse Hessian, plus a damping term, and estimating the compute cost, which comes to roughly 19 GFLOPs.

Now I'm estimating the per-step cost of the OBS updates, roughly 10ms each over 2048 steps, so about 20 seconds total. Next I'll implement it and compare against variance-ranked selection with ridge at m=1024 on the L1 and L3 holdouts.



More data helps only slowly here, so I'm testing greedy backward elimination against the variance-ranked ridge baseline on the same held-out split.



## tool_use: run_python

```json
{
 "code": "import numpy as np, time\nexec(open('builder.py').read())\niscls = np.load('ffn_in_L0_iscls.npy'); seq = np.cumsum(iscls)-1\nte = seq%4==3; tr=~te\ndef obs_select(H, W2, m, lam_rel):\n    n, F = H.shape\n    Ha = np.c_[H, np.ones(n)]\n    G = Ha.T@Ha\n    lam = lam_rel*np.trace(G[:F,:F])/F\n    G[np.arange(F),np.arange(F)] += lam\n    Hinv = np.linalg.inv(G)\n    Wc = np.c_[W2, np.zeros(W2.shape[0])].astype(np.float64)  # 768 x F+1\n    alive = np.ones(F+1, bool)\n    for step in range(F-m):\n        d = np.diag(Hinv)\n        cost = (Wc**2).sum(0)/d\n        cost[~alive] = np.inf; cost[F] = np.inf\n        i = np.argmin(cost)\n        Wc -= np.outer(Wc[:,i], Hinv[i,:]/Hinv[i,i])\n        Hinv -= np.outer(Hinv[:,i], Hinv[i,:])/Hinv[i,i]\n        alive[i]=False; Wc[:,i]=0; Hinv[i,:]=0; Hinv[:,i]=0; Hinv[i,i]=1\n    keep = np.where(alive[:F])[0]\n    return keep, Wc[:,keep], Wc[:,F]   # new W2 cols, bias delta\ndef test_obs(L, m, lam_rel):\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64)\n    A = ffn_data(L).astype(np.float64); H = gelu(A@W1.T+b1); Y = H@W2.T; Z = A+Y\n    t=time.time()\n    keep, Wk, db = obs_select(H[tr], W2, m, lam_rel)\n    pred = H[:,keep]@Wk.T + db\n    err = Y-pred\n    den = np.sqrt(((Z[te]-Z[te].mean(0))**2).sum(1)).mean()\n    return np.sqrt((err[te]**2).sum(1)).mean()/den, np.sqrt((err[tr]**2).sum(1)).mean()/den, time.time()-t\nfor L in [1,3]:\n    for lr in [1.0, 0.3, 0.1]:\n        print(L, lr, test_obs(L,1024,lr))"
}
```


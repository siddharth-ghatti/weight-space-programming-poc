## reasoning summary

I'm checking how the layer errors compare against the output variance, which has the mean removed. Every layer lands around 2–3% error at m1536, though layer 3's relative FFN error is small (about 0.0115) while its output error is larger (0.028). I'm also weighing whether that output-variance comparison holds up, since the error seems spread fairly evenly across layers.

Next I'm checking whether the errors are data-limited: I'll compare train error against test error at m1536 with 17k training tokens, and if train error is much lower, more data or a higher regularization weight should help. I'm also planning to sweep the lam values {0.01, 0.05, 0.2, 1.0} for L1 and L2 at that width.



Error is spread evenly across layers, with none acting as a bottleneck. I'm next checking the fit's data limits and whether the ridge strength of 0.05 is well chosen, using L1 and L2 at m=1536.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nrng=np.random.default_rng(0)\nfor L in [1,2]:\n    A, S = all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S); te_seq=rng.choice(useq, len(useq)//5, replace=False)\n    te=np.isin(S, te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); F=H@W2.T\n    m=1536\n    sc=H[tr].var(0)*(W2**2).sum(0); order=np.argsort(-sc)\n    keep=np.sort(order[:m]); drop=order[m:]\n    T=H[:,drop]@W2[:,drop].T; Xk=H[:,keep]\n    mu=Xk[tr].mean(0); Xc=Xk[tr]-mu; G=Xc.T@Xc; Rhs=Xc.T@(T[tr]-T[tr].mean(0))\n    denT=((F[te]-F[te].mean(0))**2).sum(); denTr=((F[tr]-F[tr].mean(0))**2).sum()\n    # also data size curve: use half of train\n    for lam_rel in [0.003,0.01,0.03,0.1,0.3]:\n        B=np.linalg.solve(G+lam_rel*np.trace(G)/m*np.eye(m), Rhs)\n        Ete=T[te]-((Xk[te]-mu)@B+T[tr].mean(0)); Etr=(T[tr]-T[tr].mean(0))-Xc@B\n        print(L, lam_rel, 'test %.4f train %.4f'%((Ete**2).sum()/denT, (Etr**2).sum()/denTr), 'n_tr', tr.sum(), flush=True)\n"
}
```


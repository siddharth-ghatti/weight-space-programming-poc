## reasoning summary

I'm noting that the linear units add nothing, since the compensation already covers the linear part through the kept units. L1's error relative to FFN output variance is much higher than L3's, so layers 1 and maybe 0 look like the hard ones. I want per-layer error at m1024 and m1536 across all layers to find where the noise comes from, measured against total FFN output variance. I'm also wondering how much the relative significance depends on the size of the FFN output.

I'm weighing whether the FFN output's size relative to the residual stream matters more than the raw error, so I'm working out the error after LN2 normalization per layer. Layers 0 and 1 are the hard ones, and since the width is uniform I can't allocate more to them, which makes me wonder whether their outputs are mostly token-identity lookups I could exploit somehow.

I'm wondering which layers actually drive the final logits, since layers 0 and 1 carry most of the error. I'll measure holdout error at m=1024, 1280, and 1536 across the five layers in a single pass, with the relative magnitude to the residual stream included.



Linear units didn't help, since holdout error rises as r grows. At m=1024, L1's holdout error is 19% of its FFN output variance versus 2.3% for L3; next I'll profile each layer across several widths to find the bottleneck.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nimport time\nrng=np.random.default_rng(0)\nres={}\nfor L in range(5):\n    A, S = all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S); te_seq=rng.choice(useq, len(useq)//5, replace=False)\n    te=np.isin(S, te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\")\n    g2=base(pre+\"output.LayerNorm.weight\")\n    H=gelu(A@W1.T+b1); F=H@W2.T\n    U=A[te]+F[te]+b2; sig=U.std(1,keepdims=True)\n    hout=(U-U.mean(1,keepdims=True))/sig*g2\n    hvar=((hout-hout.mean(0))**2).sum()\n    sc=H[tr].var(0)*(W2**2).sum(0); order=np.argsort(-sc)\n    for m in [1024,1280,1536,2048]:\n        keep=np.sort(order[:m]); drop=order[m:]\n        T=H[:,drop]@W2[:,drop].T; Xk=H[:,keep]\n        mu=Xk[tr].mean(0); Xc=Xk[tr]-mu; G=Xc.T@Xc; lam=0.05*np.trace(G)/m\n        B=np.linalg.solve(G+lam*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n        E=T[te]-((Xk[te]-mu)@B + T[tr].mean(0))\n        # error after LN2 (linearized): g2*(E - mean(E))/sig\n        Eo=(E-E.mean(1,keepdims=True))/sig*g2\n        res[(L,m)]=(((E**2).sum()/((F[te]-F[te].mean(0))**2).sum()), (Eo**2).sum()/hvar)\n        print(L,m, 'relF %.4f  rel_hout %.5f'%res[(L,m)], flush=True)\n"
}
```


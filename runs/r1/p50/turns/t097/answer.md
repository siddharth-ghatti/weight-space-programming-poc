## reasoning summary

The learning curve is flattening: going from about 12.6k to 16.9k samples only dropped the error by around 3%, so doubling the data might buy another 5-7%, which isn't a big lever since the data is near saturation. At a fixed m the error is essentially set by the selection method, so the real decision is m. I'm now wondering whether a better selection method could help with roughly 21k samples, and I'm starting to test a greedy OBS-like forward selection.

I'm checking whether the notes already settle the selection question: they say OBS-style greedy selection did no better than variance ranking on holdout, so I'll take that as given. Next I want to test a variant that weights each W2 row by the output LayerNorm gain, scoring each neuron by its variance times the squared gain-weighted outgoing weights, and run that quickly on holdout across all layers.



Extra probe data is unlikely to help much, since going from 12.6k to 16.9k tokens cut L1's holdout error only about 3%. I'm next comparing a ranking that weights units by the LN2 gain after the output LayerNorm against the current one, using post-LN holdout error.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nrng=np.random.default_rng(3)\nfor L in range(5):\n    A,S=all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S,te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); F=H@W2.T; U=A+F+b2; sig=U.std(1,keepdims=True)\n    isig2=(1/sig[tr]**2).mean()\n    out=[]\n    for name,sc in [('var',H[tr].var(0)*(W2**2).sum(0)), ('g2',H[tr].var(0)*((g2[:,None]*W2)**2).sum(0)),\n                    ('g2sig', ((H[tr]-H[tr].mean(0))**2/sig[tr]**2).mean(0)*((g2[:,None]*W2)**2).sum(0))]:\n        m=1536; order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n        T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X[tr].mean(0); Xc=X[tr]-mu; G=Xc.T@Xc\n        B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n        E=T[te]-((X[te]-mu)@B+T[tr].mean(0)); Eo=(E-E.mean(1,keepdims=True))/sig[te]*g2\n        out.append(f'{name} {(Eo**2).sum(1).mean():.3f}')\n    print(L, out, flush=True)\n"
}
```


## reasoning summary

I'm seeing that the g2sig change helps at layers 3 and 4 and is neutral elsewhere. Layer 4 seems to gain because some tokens have small σ, which amplifies them after layer norm. I'm also considering weighting the fit per token by 1/σ² so it targets post-LN error, and removing the per-token mean of the error since layer norm discards it.

I'm checking whether centering the error across hidden dims changes anything. Since each output maps independently, projecting out the per-token mean gives the same solution after projection, so it doesn't alter the error. I'll also try the weighted fit with per-token weights 1/σ², and a combined version with the g2sig selection.



Ranking by post-LN impact helps L3 and L4 notably and is neutral elsewhere. Next I'll weight the fit by 1/σ² per token and test that together with g2sig selection.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nrng=np.random.default_rng(3)\nfor L in range(5):\n    A,S=all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S,te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); F=H@W2.T; U=A+F+b2; sig=U.std(1,keepdims=True)\n    w=(1/sig[tr]**2); \n    sc=((H[tr]-H[tr].mean(0))**2*w).mean(0)*((g2[:,None]*W2)**2).sum(0)\n    m=1536; order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n    T=H[:,drop]@W2[:,drop].T; X=H[:,keep]\n    out=[]\n    for lam in [0.03,0.1,0.3]:\n        for weighted in [False, True]:\n            ww = w if weighted else np.ones_like(w)\n            mu=(X[tr]*ww).sum(0)/ww.sum(); mt=(T[tr]*ww).sum(0)/ww.sum()\n            Xc=X[tr]-mu; G=Xc.T@(Xc*ww); \n            B=np.linalg.solve(G+lam*np.trace(G)/m*np.eye(m), (Xc*ww).T@(T[tr]-mt))\n            E=T[te]-((X[te]-mu)@B+mt); Eo=(E-E.mean(1,keepdims=True))/sig[te]*g2\n            out.append(f'lam{lam} w{int(weighted)} {(Eo**2).sum(1).mean():.3f}')\n    print(L, out, flush=True)\n"
}
```


## reasoning summary

With the true statistics, the importance scores are more concentrated: keeping the top 1536 neurons preserves 92–95% of the variance, against about 82% with the proxies, and the top 1024 keep roughly 84–90%. The overlap with the old ranking is only 0.62–0.79, so the old ranking was clearly off. I'm also going to check the linear-unit compensation next, computing the matrix for the pruned set and seeing how much variance it accounts for relative to the total.

Now I'm working out the linear-unit compensation: the pruned set's linear part can be folded into a matrix whose variance I can compute directly, so I'll weigh how many units to prune against the nonlinear and linear error terms for each choice of rank.



With true statistics, the top 1536 units keep 92–95% of output variance, and the old proxy ranking overlapped the real one only 62–79%, so the proxy caused much of experiment A's damage. Next I'll measure the linear-unit compensation by implementing the top singular directions of the pruned linear parts and comparing residual error against plain top-m selection.



## tool_use: run_python

```json
{
 "code": "def analyze(l, m, rs=(0,32,64,128,256)):\n    d=np.load(f'timp_L{l}.npz'); I,NL,al,cov=d['I'],d['NL'],d['al'],d['cov']\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n    tot=I.sum()\n    # plain: keep top-m by I, error = sum pruned I (independence approx)\n    keepI=np.argsort(-I)[:m]; prI=np.setdiff1d(np.arange(3072),keepI)\n    # with cross terms for linear part: pruned linear var exact\n    out={'plain_indep':I[prI].sum()/tot}\n    ev,evec=np.linalg.eigh(cov); ev=np.maximum(ev,0); S12=evec*np.sqrt(ev)\n    for r in rs:\n        k=m-r; keep=np.argsort(-NL)[:k]; pr=np.setdiff1d(np.arange(3072),keep)\n        M=(U[:,pr]*al[pr])@W1[pr]           # 768x768\n        MS=M@S12; sv=np.linalg.svd(MS,compute_uv=False)\n        linvar=(sv**2).sum(); resid_lin=(sv[r:]**2).sum()\n        out[r]=(NL[pr].sum()+resid_lin)/tot\n    out['plain_linvar_exact']=None\n    return out\nfor l in range(6):\n    for m in [768,1024,1536]:\n        o=analyze(l,m)\n        print(f'L{l} m={m}: plain(indep) {o[\"plain_indep\"]:.3f} | NL-select + r linear units: '+' '.join(f'r{r}:{o[r]:.3f}' for r in (0,32,64,128,256)))"
}
```


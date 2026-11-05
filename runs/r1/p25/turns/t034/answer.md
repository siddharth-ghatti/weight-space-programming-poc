## reasoning summary

I'm noting that the "plain(indep)" comparison uses an independence approximation for the linear parts too, so it isn't a like-for-like match. I want a fairer baseline, so I'll compute the plain error with exact linear-part cross terms over the pruned set. Even so, the linear units clearly help layer 5, dropping from 0.236 to 0.098 at m=768, and they help moderately in the other layers. I'm also wondering how accurate that estimate actually is.

I'm asking whether the error fraction actually predicts downstream fidelity, and I'd like to check it against experiment A's selection under the true statistics, since that run hit 91.7% agreement. I'm also considering normalizing the error against the residual output variance and the total variance of the activations, to see how it looks from that angle.



Linear units cut the estimated error a lot in layer 5 (0.24 to 0.10 at m=768) and help moderately elsewhere. Next I'll score experiment A's selection (91.7% agreement) and plain top-m selection by true importance with the same measure, to check the estimates.



## tool_use: run_python

```json
{
 "code": "def err_for(l, keep, r=0, mode='exact'):\n    d=np.load(f'timp_L{l}.npz'); I,NL,al,cov=d['I'],d['NL'],d['al'],d['cov']\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n    pr=np.setdiff1d(np.arange(3072),keep)\n    ev,evec=np.linalg.eigh(cov); S12=evec*np.sqrt(np.maximum(ev,0))\n    M=(U[:,pr]*al[pr])@W1[pr]; sv=np.linalg.svd(M@S12,compute_uv=False)\n    trA=np.trace(cov)\n    e=NL[pr].sum()+(sv[r:]**2).sum()\n    return e/I.sum(), e/trA\nprint('exp A selection (proxy top1536, no linear units):')\nfor l in range(6):\n    keep=np.argsort(-np.load(f'imp_L{l}.npz')['I'])[:1536]\n    print(f'  L{l}: err/FFNvar {err_for(l,keep)[0]:.3f}  err/tr(cov a) {err_for(l,keep)[1]:.4f}')\nprint('true-stat plain top-m by I (exact linear cross terms):')\nfor m in [1024,1536,2048]:\n    print(m,' '.join(f'L{l}:{err_for(l,np.argsort(-np.load(f\"timp_L{l}.npz\")[\"I\"])[:m])[0]:.3f}' for l in range(6)))"
}
```


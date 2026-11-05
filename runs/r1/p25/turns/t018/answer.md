## reasoning summary

The layer 1 CLS relative error is 0.096, well above the 0.013 spread across samples, so the edit is systematically shifting the CLS token by about 10%. My guess is that the layer 0 bias compensation, estimated from all-token statistics, doesn't suit CLS, which has its own distribution. Mean token error at layer 1 is also high at 0.135.

I'm noting that Layer 0's FFN pruning looks very damaging, and its proxy input is weak since the attention step changes the embeddings a lot. The `_gh_x` helper isn't exported under `from ffnlib import *`, so I'll import it explicitly. Getting true FFN inputs would help, but six extra passes is expensive with 22 left.

I'm weighing whether the extra passes to get true FFN inputs are worth the cost, given only 22 remain. Probing two layers in one pass doesn't work, since the earlier layer has to stay intact for the later one to get correct input. So the open question is whether better importance estimates and bias compensation justify it. The variance breakdown should show how much the linear compensation actually helps, so I'll fix the import and run that first.



Layer 0's pruning shifts layer-1 CLS by about 10% relative, versus only 1.3% variation across inputs, and the embedding-output proxy is weak there because layer-0 attention reshapes the representation. I'm fixing the import bug, then running the nonlinear-variance analysis.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK)); import ffnlib as F\nfrom scipy.stats import norm as N\nmus=np.load('tok_mu.npy'); covs=np.load('tok_cov.npy')\ndef lin_coef(m,s):\n    z=m[:,None]+s[:,None]*F._gh_x[None,:]\n    dg=N.cdf(z)+z*N.pdf(z)\n    return (dg*F._gh_w).sum(1)\nfor l in range(6):\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    b1=base(f'roberta.encoder.layer.{l}.intermediate.dense.bias').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64); un=(U**2).sum(0)\n    tot=[];nl=[];lin=[]\n    for src in [l,l+1]:\n        m,s=F.unit_stats(W1,b1,mus[src],covs[src]); Eg,Vg=F.gauss_gelu_moments(m,s); al=lin_coef(m,s)\n        tot.append(Vg*un); nl.append(np.maximum(Vg-al**2*s**2,0)*un); lin.append(al)\n    T=np.sqrt(tot[0]*tot[1]); NL=np.sqrt(nl[0]*nl[1])\n    s_=np.sort(NL)[::-1]\n    print(f'L{l}: NL share {NL.sum()/T.sum():.3f}; top-k NL kept: 512 {s_[:512].sum()/NL.sum():.3f} 1024 {s_[:1024].sum()/NL.sum():.3f} 1536 {s_[:1536].sum()/NL.sum():.3f}')"
}
```


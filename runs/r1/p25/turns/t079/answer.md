## reasoning summary

For the practice margin band of 0 to 0.5, the expected net per flip toward the second choice is about -0.25. The observed flips at m=2048 went worse than that, at -0.6 per flip over 15 flips, though the sample is small and was measured in-sample, so it could be chance. The flips only seem to happen when the margin falls below roughly 0.55.

I'm estimating how many test items fall under a margin of 0.55, since flips only happen there. If that share is around 12%, the expected net loss is roughly 3 items for the m=2048 variant and about 4.5 for C, which puts C at real risk of failing the threshold, with noise on top.

I'm weighing whether to chase better fidelity by using a larger m, or by improving the fitting method itself. The flips seem to come from upstream errors introduced by pruning the early FFN layers, and a closed-form head refit on the edited CLS gave a small cross-validated gain (0.981 versus 0.978 for C). I'm now considering whether the last-layer FFN could be fit directly to the head's logits instead of the full base representation.

I'm checking whether a last-layer FFN fit could target just the four logits through the head, rather than all 768 dimensions of the base representation. That might be an easier target, but a fit on the practice set could look better in-sample than it does on test, so it would need cross-validation before I trust it.

I'm considering whether a repeat of the layer-5 correction with a fresh activation set for the final configuration would help, and whether similar corrections at layers 1 through 4 could work despite the attention mixing making token-level errors harder to match.

I'm thinking about whether an OBS-style score, ΔE_i = ||U_i||² / (H⁻¹)_ii computed from the full 3072×3072 Gram matrix, could now work with about 14k tokens, where it failed earlier with only around 3k.

I'm weighing a few smaller tweaks, like tuning the per-layer regularization and the head weights, though those probably give little. I'm more interested in whether the last-layer FFN could be fit to the base representation at the CLS token, since that's where I have the most data, about 1000 samples. The idea is to predict the upstream error from the edited activations, and a mixed fit already reaches a cross-validated R² of 0.956, above the local fit's 0.949.

I'm weighing whether a larger m, around 2304 or 2560, would be the more reliable lever. Per-layer errors would shrink by roughly three times, which should cut the expected test flips to about 7 and the net loss to around 1.5 items, a safer margin than the current options.

I'm planning to re-test the OBS selection for the early layers using the larger dataset, then build an m=2048 candidate that includes the layer-5 mixed fit, though that fit was made for a different upstream configuration and may not transfer directly.

I'm planning to build the m=2048 candidate with the new data and the layer-5 mixed fit, then evaluate it in one pass, expecting roughly 0.988 fidelity. After that I'd compare m=2048 against a smaller m such as 1792 before settling on one.

I'm going to check whether the OBS-style selection works now that there's far more data, since it failed earlier with only about 3k tokens. If it does, it could help the early-layer fits, so I'll try it before building the m=2048 candidate.



Flips only happen below margin 0.55, and on a harder test set I estimate C would net about −4 items, too close to the limit, so fidelity is the priority. Next I'll retest OBS-style selection against data-score selection at m=1536 and 2048, now that layers 0–2 have about 14k tokens.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy'); fold=grp%4; tk2=np.load('tok2_kind.npy')\ndef cvsel(l, ms, hs=(1.,10.,100.), lam=100., wcls=20., weos=5.):\n    pre=f'roberta.encoder.layer.{l}.'\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64); a2=np.load(f'true2_a_L{l}.npy').astype(np.float64)\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2; G2=gelu(a2@W1.T+b1)\n    num={}; den=0\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        Gt=np.vstack([G[tr],G2]); kt=np.concatenate([tk[tr],tk2])\n        w=np.where(kt==0,wcls,np.where(kt==1,weos,1.0)); sw=np.sqrt(w)[:,None]\n        mu=(w[:,None]*Gt).sum(0)/w.sum(); Gc=(Gt-mu)*sw\n        H=Gc.T@Gc; un=(U**2).sum(0)\n        scores={'data':un*np.diag(H)}\n        for h in hs:\n            Hi=np.linalg.inv(H+h*np.eye(3072)); scores[f'obs{h:g}']=un/np.diag(Hi)\n        Fm=F[tr][tk[tr]==2].mean(0); den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n        for m in ms:\n          for name,sc in scores.items():\n            keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n            A=H[np.ix_(keep,keep)]; B=H[np.ix_(keep,pr)]@U[:,pr].T\n            D=np.linalg.solve(A+lam*np.eye(m),B).T\n            Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n            e=((Fe-F[te])**2).sum(1)\n            num[(m,name)]=num.get((m,name),0)+e[tk[te]==2].sum()\n    return {k:round(v/den,4) for k,v in num.items()}\nprint(0, cvsel(0,(1536,2048)))\n"
}
```


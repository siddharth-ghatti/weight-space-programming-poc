## reasoning summary

I'm looking at the margins: the median is 2.83, and about 12% fall under 1. Accuracy in the under-1 bin is 0.705 on practice, probably similar or lower on test, where overall accuracy is 77.5%, so test margins are likely a bit smaller, maybe a median near 2. Next I'm estimating how many items could flip under the logit perturbation at m=2048.

I'm estimating the test flip rate as about 2-3%, roughly 8-12 items out of 400. But for items right at the decision boundary the two classes are about equally likely, so flips there may roughly cancel out on average, which would mean the expected net loss is close to zero.

The expected net loss from the flips looks close to zero, since at such small margins the top two classes are roughly equally likely, so it's mostly variance. With about 10 flips the standard deviation is around 2 to 3 items, and losing 5 or more would break the 1% budget, which I put at roughly a 2 to 5% chance. Macro-F1 drops about 0.02, and a single governance item moves that class's F1 noticeably but the macro effect stays small. So m=2048 looks reasonably safe, and m=1536 would roughly double the perturbation and push the flip count up to around 1.

I'm weighing the m=1536 option: its perturbation would be roughly 0.25 at the median, which means about 18 test flips and a 10-15% chance of breaking the 1% budget. That's riskier than m=2048, so I'd want to spend some passes improving the lower-m method before committing to it.

I'm sketching out the next steps: first running sequential error-feedback probes for layers 1 through 5 at m=1536, then a final verification pass, with a few spare passes kept aside, or alternatively spending those passes on extra data for the lower layers.

I'm checking the probe setup for this layer: the FFN pruning at m units is fine since the layer's FFN output is zeroed, so the kept units don't matter. I'm also wondering whether one probe pass could read more than one layer, but a layer's output feeds the next layer, so each pass can only handle one layer. Then I'm considering whether to leave the FFN intact and read a different quantity from the layer instead.

I'm weighing whether the inversion route is worth it: from the edited layer output I could try to back out the attention output by solving a + f(a) = x with a fixed-point iteration. That's a nonlinear solve over 768 dimensions plus two scalars per token, which feels more like a new method than the closed-form fitting I've been doing.

I'm considering whether to invert the FFN instead: the layer's output x is known, and since the layernorm mean and scale are unknown, I'd solve for the pre-norm attention output by Newton iteration over the 768 dimensions plus the two scalars. Convergence is uncertain because the FFN output is large, so this would be a new nonlinear method rather than the closed-form fitting I've been doing.

I'm weighing whether sequential correction at layer 1 is worth it, since L0's error leaks into L1's FFN target and that FFN is pruned, so it may not fit the undo well. Given that, the gain could be modest, so I'm considering whether spending the same passes on more base data for the lower layers would do more.

I'm weighing the cost of different model sizes against how accurate each stays, and I'm considering layer-drop at m=2048 as a comparable alternative to m=1536. Layer-dropping lost a lot of accuracy earlier, so I'm leaning against it, and I'd like to start by testing m=1536.

I'm going to test the m=1536 setup as it stands before changing anything, to see how much accuracy it loses. Then I'll tune the regularization strength between 30 and 300, and check how the CLS and EOS weights affect the layer 5 fit.

I'm about to run the offline cross-validation at m=1536 to tune the ridge strength and the CLS weighting, since the head-relevant output directions might be where the fit needs help.

I'm going to check whether m=1536 holds up better with a different ridge strength, so I'm sweeping λ across 30, 100, 300, and 1000 on layers 0 through 4, using the same data-selection approach as before.



Margins are thin on practice items, but the m=2048 perturbation is small, so flips should be near coin-tosses and test loss low. Next I'll test whether m=1536 is reachable by tuning λ on held-out folds for layers 0–4.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy')\nfold=grp%4\ndef cv(l, m, lams, wcls=20.0, weos=5.0):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2\n    num={lam:np.zeros(3) for lam in lams}; den=0\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        w=np.where(tk[tr]==0,wcls,np.where(tk[tr]==1,weos,1.0)); sw=np.sqrt(w)[:,None]\n        Gt=G[tr]; mu=(w[:,None]*Gt).sum(0)/w.sum(); Gc=(Gt-mu)*sw\n        sc=(U**2).sum(0)*(Gc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n        A=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n        Fm=F[tr][tk[tr]==2].mean(0); den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n        for lam in lams:\n            D=np.linalg.solve(A+lam*np.eye(m),B).T\n            Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n            e=((Fe-F[te])**2).sum(1)\n            for k in range(3): num[lam][k]+=e[tk[te]==k].sum()\n    return {lam:(num[lam]/den).round(4) for lam in lams}\nfor l in range(5):\n    print(l, cv(l,1536,[30,100,300,1000]))\n"
}
```


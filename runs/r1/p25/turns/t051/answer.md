## reasoning summary

I'm treating the denominator as the normal-token mean deviation for every kind, so the k0 numbers look large but still work for comparison. For the normal-token errors at ridge λ=100, L0 drops from 0.091 to 0.042 to 0.012 across the three widths, L1 gives 0.13, 0.063, 0.021, and L2 gives 0.068 and 0.035. I might also try λ=1000.

Now I'm weighing whether data-based importance selection is worth it, since the Gaussian importance estimate predicted lower error but the actual error came out higher. I want to test ridge values of 300, 1000 and 3000, and check whether λ should scale with the trace.

I'm comparing the layer errors from pass 11 with the new LS numbers at m=2048 and λ=100. Summed over layers, the error drops to about 0.148 from roughly 0.36, so if disagreement scales with the square root of total error, I'd expect around 5%, which still isn't good enough. The CLS error in pass 11 may have been the bigger culprit.

Sizing up the trade-offs now: at m=2560 the summed layer errors fall to about 0.048, which suggests roughly 3% disagreement, though that square-root extrapolation is rough. The saving is only around 4.7M parameters, so the next question is which FFN units are cheapest to drop in accuracy per parameter, especially in layers 3 and 4.

I'm wondering whether dropping an entire layer, like layer 3 or 4, would beat trimming units, but its attention would still be needed. Another idea is folding a hard layer's FFN units into spare capacity in a cheaper layer, though those units act on different attention outputs, so that probably won't work.

I'm weighing whether a layer-to-layer hand-off could work, but the FFN outputs feed into the next layer's attention, so moving that computation would change the attention inputs and wouldn't be exact. Turning to the test set instead: it's from the same distribution as training, with a base accuracy of 77.5%, so the tolerable drop is about one point, roughly four of the 400 examples.

I'm estimating how many flips the test set could absorb, roughly four net errors out of 400. If flips split evenly between good and bad, the net stays near zero, but a 60/40 bias toward bad flips at 5% disagreement would likely push past the threshold. The in-sample base accuracy looks much higher than the test's 77.5%, so I'm wondering whether the practice labels could help pick which units are safe to cut.

I'm running the parameter counts for the m=2304 option, multiplying out the per-layer size to see how much it saves against the baseline.

I'm checking the parameter totals for the wider options: m=3072 comes to about 82.3M, and m=2560 to about 77.6M. Dropping one layer at m=3072 gives roughly 75.2M, about the same as m=2304 with all six layers kept, and at m=2560 with five layers the total is around 71.3M.

I'm weighing whether dropping the last layer could work better than trimming units, since its CLS attention pooling would be lost. The refit for dropping layer 5 came out around 0.955, which is in the same ballpark as the FFN pruning estimate, so I'm considering whether combining the two cuts could help.

I'm considering whether to accept a modest reduction, since the cleaner options look limited. Pruning FFN units could still improve by choosing units in a least-squares-aware way, or by collecting more probe data for the harder early layers to reduce overfitting. I'm also thinking about weighting errors by how much the downstream layers actually use each direction, though that gets complicated.

I'm checking the error relative to the FFN output's own variation, since the residual gets layer-normalized afterward, and the per-token direction is what counts. I also want to see whether errors compound through the later layers, so a single calibration point at another width would help before I run a least-squares test at m=2048.

I'm going to try a few refinements offline before testing: comparing unit selection by Gaussian importance, by data score, and by a combination, along with larger ridge values. I'm also considering adding a few linear units, since they could capture the linear part of the pruned units' behavior, though the FFN doesn't expose its input directly, so that would need a regression on the projected input.

I'm weighing whether the linear part of the FFN output could be folded into the next step, but the layer-norm and attention structure means it can't be absorbed cleanly, since the input to the layer is only a diagonal gain away from the normalized residual.

I'm considering adding linear units to the least-squares fit, so each selected direction gets a regression feature that stands in for its pruned contribution. That would mean setting each linear unit's pre-activation far enough into GELU's linear range that its output is effectively linear in the input, then folding the fitted coefficients back into the output weights.

I'm going to fit the pruned units' contribution with linear features: pick the input directions where the pruned residual is most linearly predictable, then run a joint least-squares fit over the kept units and those features, and check it with cross-validation.

I'm going to sweep ridge values of 100, 300, and 1000 at m=2048 with 0, 64, and 128 linear units, choosing the linear directions inside each training fold so the cross-validation stays honest.



Least-squares refitting clearly helps, cutting the error on special tokens to near zero and ordinary tokens by 1.5–3×. Next I'll try larger ridge values and add linear units as extra regressors, scoring everything on held-out folds at m=2048.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy')\nfold=grp%4\ndef run2(l, m, r_list, lams, wcls=20.0, C=15.0):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2\n    out={}\n    for r in r_list:\n      for lam in lams: out[(r,lam)]=[0,0]\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        w=np.where(tk[tr]==0,wcls,1.0); sw=np.sqrt(w)[:,None]\n        Gt=G[tr]; mu=(w[:,None]*Gt).sum(0)/w.sum()\n        sc=(U**2).sum(0)*((w[:,None]*(Gt-mu)**2).sum(0)/w.sum())\n        order=np.argsort(-sc)\n        amu=(w[:,None]*a[tr]).sum(0)/w.sum()\n        Fm=F[tr][tk[tr]==2].mean(0)\n        dte=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n        for r in r_list:\n            keep=np.sort(order[:m-r]); pr=np.setdiff1d(np.arange(3072),keep)\n            R=((Gt[:,pr]-mu[pr])@U[:,pr].T)\n            ac=(a[tr]-amu)\n            if r>0:\n                # linear predictability of R from a\n                Bl=np.linalg.solve((ac*sw).T@(ac*sw)+1e-1*np.eye(768),(ac*sw).T@(R*sw))\n                Fit=(ac*sw)@Bl\n                _,_,Vt=np.linalg.svd(Fit,full_matrices=False)  # Fit = U S Vt ; Vt in output space\n                # input directions: Bl @ Vt.T\n                V=Bl@Vt[:r].T  # 768 x r\n                V=V/np.sqrt(((ac@V)**2).mean(0))  # unit std\n                Ltr=ac@V; Lte=(a[te]-amu)@V\n            else:\n                Ltr=np.zeros((tr.sum(),0)); Lte=np.zeros((te.sum(),0))\n            X=np.hstack([Gt[:,keep]-mu[keep],Ltr])*sw\n            A=X.T@X; Bm=X.T@(R*sw)\n            for lam in lams:\n                reg=lam*np.ones(m); reg[m-r:]=1e-3\n                D=np.linalg.solve(A+np.diag(reg),Bm).T\n                Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+np.hstack([G[te][:,keep]-mu[keep],Lte])@D.T\n                e=((Fe-F[te])**2).sum(1)\n                out[(r,lam)][0]+=e[tk[te]==2].sum(); out[(r,lam)][1]+=e[tk[te]==0].sum()\n            out_den=dte\n        for k in out: pass\n        if f==0: dens=0\n        dens+=dte\n    return {k:(round(v[0]/dens,4),round(v[1]/dens,5)) for k,v in out.items()}\nfor l in (0,1,2):\n    print(l, run2(l,2048,[0,64,128],[100,300,1000,3000]))\n"
}
```


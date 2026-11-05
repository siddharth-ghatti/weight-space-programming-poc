## reasoning summary

I'm seeing that OBS with h=1 gives only a small improvement, about 5%. I want to try smaller values of h like 0.1 and 0.3, and also test a combined score that mixes the data and obs measures. Those tweaks look like they'll only yield marginal gains, so I'll need a bigger idea. I'm weighing what limits performance when only the output weights are refit, and whether letting the kept units' input weights change could help.

I'm considering whether the attention weights could compensate for pruning errors in the FFN, since attention can't be pruned itself. The nonlinear nature of the error makes a simple linear fix unlikely to work, so I'm weighing other options like diagonal scaling in the layer norms and treating the hardest layers separately.

I'm thinking about whether the uniform-m constraint is the real bottleneck, since layers 3 to 5 need far fewer units than layers 0 to 2. One idea is to move part of a hard layer's FFN work into the next layer's slots, though that would make those units see a different input than the one they were fit on.

I'm considering whether to drop the FFN of layer 4 altogether, but since m is uniform across layers I can't zero out just one layer's FFN. So I'm going to accept the FFN error and work with the remaining options.

I'm now checking whether the capacity in layers 3 to 5 could correct errors coming from layers 0 to 2, though the upstream error seems hard for a small FFN to represent. I'm also weighing m=1792 or 2048 for better reliability, and wondering whether my estimates of test flip rates are too pessimistic.

I'm estimating that test flips at the low-margin items come to roughly 3-4.5% for m=2048 and 4.4-6.7% for C, and I'm checking whether the per-flip net is positive or negative given the observed flip ratios, which look worse than the practice estimate.

I'm rechecking the sign of the per-flip net, since the margin-based estimate leads to a pessimistic result. Even at m=2048, the estimated loss looks like it could be sizable, but that depends on whether top-1 really beats top-2 on held-out low-margin items, which the in-sample numbers may overstate.

Checking the sign again: if each flip nets only about -0.05 to -0.1, then m=2048 expects roughly one to two items lost, and the chance of losing five or more is around 10-15%.

I'm estimating the flip counts for m=2048 and C at roughly 20-25% failure odds, and I'm noting that the noise from each flip dominates. Since fewer flips is what I need, I'm weighing m=2560, where the per-layer error drops to about 0.58 of the m=2048 level, giving around 9 flips and a failure chance near 3-5%.

I'm thinking a smaller m around 2304 might be a reasonable middle ground, trimming about 9% off the size. I also wonder whether refitting the final head could cut the flips, though the gain looks small. Since flips cluster near zero margin, the only real fix is a closer reproduction of the perturbation.

I'm looking at whether I can cancel the layer-0 FFN error exactly, since that layer seems to contribute most of the perturbation to the final logits. To pin down which layer matters most, I'm thinking about building a diagnostic that zeros out pruned units in selected layers while keeping the structure at a larger m.

I'm wondering whether the token-specific units in layer 0 could have their effect carried by the value path of the attention, since the embeddings are frozen and can't absorb it directly. Layer 0's input depends mostly on token identity and position, so a unit that fires for certain tokens adds a nearly fixed vector, which might be representable through the value projection.

I'm wondering whether the output can be made to cancel the pruning error directly, since the hidden units only ever enter through the kept rows of the first layer. That would mean refitting those rows with a closed-form correction rather than fitting the outputs alone.

I'm weighing whether to replace a kept unit with a merged unit built from a cluster of similar original units, which is essentially what the least-squares redistribution already does implicitly. I think the realistic gains are incremental, so the main decision is the size m, and I want to compare 2048, 2304, and 2560 using per-layer errors from the new OBS selection and the two earlier calibration points.

Estimating the flip counts for m=2048 with the new OBS selection, I get around 13 flips, scaling the earlier pass by the lower per-layer errors. Next I'm working out how the m=2304 candidate compares.

Comparing the sqrt-scaling estimates: m=2560 looks to bring flips down to around 8, while m=2304 lands near 10, so the cost of a smaller model against the test-flip risk is the trade-off I'm still weighing.

I'm working out the failure odds at each size, assuming a 400-item test set and roughly double the practice flip rate. That puts m=2048 near 11 flips, the C setting near 18, and m=2560 near 6. With the flip outcomes split roughly evenly between losses and gains, the expected net loss scales with the flip count, so I'm checking how often it reaches five or more.

I'm checking the expected loss at each size: with a loss of five or more items failing the threshold, m=2560 comes out around 4.5%, m=2048 around 12.5%, and the C setting around 23%. If the per-flip net is closer to 0.25 instead, the picture shifts, and I'm working through what m=2048 looks like under that assumption.

I'm checking whether any FFN units are exactly dead or exactly linear over the calibration data, since pruning or merging those would add no error. A sum of linear units is still linear, so merging them doesn't obviously reduce the unit count without changing the function.

I'm checking whether any units are dead or nearly linear on the calibration data, since merging those would add no error. Even so, linear units only compress when their count exceeds the hidden width, so the gain looks small. The real question is how much error budget a smaller m can afford.

I'm checking whether flipping the head bias to match the base class prior would cut the systematic shifts, though the flip counts look roughly balanced already. Weighing the trade-off: m=2560 keeps failure risk around 4-5%, m=2048 near 12%, and m=1536 near 23%, against sizes of about 77.6M, 72.9M, and 68.1M parameters.

I'm considering whether a sharper error estimate could close the gap at m=2048 without shrinking further. Tighter least-squares fitting, extra calibration data, and a targeted correction on the final head might together cut the error by roughly a quarter, which would still leave the flip count well above where m=2560 sits. I'm also wondering whether the flip-relevant behavior depends only on the CLS output, which would narrow which layers need the most care.

Tiny-margin items are close to coin flips on the test set anyway, so a flip there costs roughly nothing in expectation. The real worry is the flips at moderate margins, and that's where the per-flip net loss actually matters.

Each flip at a tiny margin is nearly a coin toss on the test set, so it costs little in expectation and mostly adds variance. Flips at larger margins are the ones that really hurt, since the calibrated probability gap there is about 0.2 per flip. Because the spread grows with the square root of the flip count, the failure risk is driven mainly by variance, so I'm checking how far the 4.5-point threshold sits from the expected loss at m=2048.

I'm checking whether the test-set size and the flip multiplier are actually known, since my estimate of the failure risk depends on both. I'm trying to recall the TCFD recommendations dataset's split sizes, though I'm not certain of them.

I'm checking the test split size, since my dataset recollection is shaky: 400 items would make each point worth four items, while 320 would make each point worth about 3.2 and leave less room to absorb flips. Either way, the fail risk at m=2048 and the smaller sizes looks substantial enough that I'm leaning toward a more conservative size like 2304 or 2560, or spending more calibration passes to tighten the fidelity first.

Next I want to check the test split size and how many low-margin items it likely has, since the failure odds hinge on both. Practice accuracy alone doesn't tell me the test margin distribution, so I'm assuming the test set carries somewhere around 1.5 to 2.5 times the low-margin density. That makes it worth spending the remaining passes on fidelity before committing to a size, starting with offline improvements.

I'm planning to tune the OBS step size and the regularization on the new data, possibly per layer. Then I'll build an improved m=2048 candidate with the extra augmentation, evaluate it in one pass, and decide between 2048 and a size like 1792 or 2304 before finalizing.

I'm looking at whether the head can be refit in the pre-tanh space, regressing the inverse-tanh of the base pooled outputs onto the edited CLS features while leaving the output projection untouched.

I'm checking whether a head refit could fix much, and it looks like the linear refit only recovered about 0.3% agreement, so that's small. I'm also considering a final-layer FFN correction: if its output could be set to match the base representation, the CLS output would line up with the base model, so I'm working out how close that could get.

The final layer's FFN seems unable to learn the correction from its own input, so the missing information must come from the attention pooling over corrupted tokens at layer 5 and earlier. I'm accepting that per-flip fidelity is close to the limit of this method, so remaining gains will be incremental. The choice of m will rest on risk, and I'll start with the offline improvements.



OBS damping with h=1 gives a small gain, and the fail risk at m=1536 and 2048 is mostly coin-toss variance on near-zero-margin items. I'm tuning h and λ jointly for layers 0–2 on the combined data before choosing m.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ngrp=np.load('tok_grp.npy'); tk=np.load('tok_kind.npy'); fold=grp%4; tk2=np.load('tok2_kind.npy')\ndef cvsel2(l, m, hs, lams, wcls=20., weos=5.):\n    pre=f'roberta.encoder.layer.{l}.'\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64); a2=np.load(f'true2_a_L{l}.npy').astype(np.float64)\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2; G2=gelu(a2@W1.T+b1)\n    num={}; den=0\n    for f in range(4):\n        tr=fold!=f; te=~tr\n        Gt=np.vstack([G[tr],G2]); kt=np.concatenate([tk[tr],tk2])\n        w=np.where(kt==0,wcls,np.where(kt==1,weos,1.0)); sw=np.sqrt(w)[:,None]\n        mu=(w[:,None]*Gt).sum(0)/w.sum(); Gc=(Gt-mu)*sw\n        H=Gc.T@Gc; un=(U**2).sum(0)\n        Fm=F[tr][tk[tr]==2].mean(0); den+=((F[te]-Fm)**2).sum(1)[tk[te]==2].sum()\n        for h in hs:\n            sc=un*np.diag(H) if h is None else un/np.diag(np.linalg.inv(H+h*np.eye(3072)))\n            keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n            A=H[np.ix_(keep,keep)]; B=H[np.ix_(keep,pr)]@U[:,pr].T\n            for lam in lams:\n                D=np.linalg.solve(A+lam*np.eye(m),B).T\n                Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n                e=((Fe-F[te])**2).sum(1)\n                num[(h,lam)]=num.get((h,lam),0)+e[tk[te]==2].sum()\n    return {k:round(v/den,4) for k,v in num.items()}\nfor l in (0,1,2):\n    print(l, cvsel2(l,2048,[None,0.1,0.3,1.0],[30,100,300]))\n"
}
```


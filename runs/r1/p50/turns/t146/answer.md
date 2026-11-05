## reasoning summary

I'm estimating failure probabilities for each model size from the donor model, with the largest sizes showing the lowest risk. I'm also noting that the baseline failure probability is zero when there's no noise, and that under calibration, flipping away from the argmax always carries expected loss even with symmetric noise.

I'm working out what k likely is: the base model is probably overconfident on the test set, so k lands around 1.0 to 1.25, and I'll take about 1.25 as the central value. That puts failure odds near 14-17% for m1536, 11-13% for m1792, and 7-8% for m2048. Under a plain linear expected score, m1536 still comes out ahead of the larger sizes, and the smaller m1024 might be competitive too, though a fail costing more than zero would change that.

I'm weighing whether the remaining passes could cut the noise at m1536 enough to matter, since the risk seems to come from low-margin items with large noise along the decision direction. Without knowing which residual-stream directions carry the decision at the earlier layers, I can't really target that compensation, so the ~15% failure estimate probably stays about where it is.

I'm weighing whether a constant bias shift along a candidate direction could reveal its downstream effect on the logits. That would take one forward pass per direction, which is far too costly across 768 dimensions and five layers. I'm also considering LN2's gamma as a proxy for which directions matter, though the earlier attempt at selecting by that signal didn't help.

I'm wondering whether I can shrink the logit noise by making the final classifier less sensitive to the directions where the noise sits. I have base and edited CLS states for each item, so the difference gives the noise per item, but the earlier attempt at a denoising head refit with ridge overfit at this dimensionality.

I'm considering whether to project out the highest-noise directions from the classifier input, which would reduce noise in those directions but also drop some signal. I'm also questioning whether refitting the classifier head offline on the base and edited hidden states counts as reimplementing the forward pass, given earlier sessions already did offline head refits.

I'm weighing whether the head-refit approach is acceptable at all, since computing the classifier on returned hidden states looks like a small, analysis-only step rather than a reimplemented forward pass. Earlier attempts at that refit gave no gain, which suggests the noise at the final CLS state doesn't sit in a cleanly separable subspace, or there aren't enough samples to pin it down. I'm also returning to the asymmetry seen on practice data: the edit tends to shrink margins for items the base already gets right and grow them for items it gets wrong.

I'm weighing whether a more conservative choice like m1792 at roughly 11% risk is worth giving up some parameter savings over m1536 at about 14%, since a failure costs more than a small gain in size. The grader's ranking of failing submissions is unclear, so I'm leaning toward the option that balances risk against reduction.

I'm considering whether a smaller reduction in noise could be enough, since matching the m1792 noise level would mean cutting variance by about 35%. The greedy selection only gave a small gain on proxies and nothing at full scale, so that route doesn't look promising. I'm also thinking about correcting upstream token-level errors at layer 4 or 5, though an earlier attempt at L5 FFN correction against base targets didn't help, so I'm not sure that's worth pursuing.

I'm going to check whether a model at m1280 is worth building, since it would take one pass to evaluate its failure odds. If its risk comes in around 25%, the expected score would land close to what m1536 already gives, so the extra check would only confirm the same tradeoff I'm already weighing.

Under a linear expected score the EV is nearly flat, so the choice comes down to risk appetite. Pricing a failure at roughly a 10% reduction still leaves m1536 on top, though the margin narrows as the penalty grows.

Under these risk estimates, m1536 still beats the larger sizes: going from 2048 to 1536 adds about 8 points of failure risk for roughly 5.7 points of reduction. Only a fail penalty of nearly half a 49% reduction would flip that, which seems extreme, so I'm leaning toward m1536 or smaller. Next I need to check whether m1280 is worth evaluating, and its noise level is the open question.

I'm checking the m1280 numbers: extrapolating the noise trend puts failure risk near 25-30%, which gives roughly the same expected score as m1536 and loses under any penalty for failing, so m1536 stays the choice. Next I'm comparing v1 and v3 on the donor simulation, where v3 shows slightly lower failure rates, though the original empirical simulation favored v1.

I'm checking whether combining the best per-layer choices could push m1536's noise down, since the holdout proxies favored greedy selection at several layers, though full-model results haven't confirmed that. I'm also considering a less noisy metric, a margin-weighted directional RMS, because the practice flips hinge on only about fifty sensitive items.

I'm splitting the low-margin items out from the rest and comparing v1 and v3 on how far their margins move in each bin, since a plain overall RMS may hide where the flips actually happen.



The m1536 donor model still comes out best under a linear utility, and I'm now choosing between v1 and v3 (possibly with a per-layer mix) using margin-change rms on the low-margin items.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); y=np.load('y.npy')\npb=bl.argmax(1); s=np.sort(bl,1); marg=s[:,-1]-s[:,-2]\ndef dmarg(lg):\n    oth=np.where(np.eye(4)[pb]>0,-np.inf,lg).max(1)\n    return (lg[np.arange(1000),pb]-oth)-marg\nM={'v1':'lg_pass25.npy','v2':'lg_pass26.npy','v3':'lg_pass29.npy','m1792':'lg_pass31_m1792v2.npy','m2048':'lg_pass30_m2048v2.npy'}\nfor n,f in M.items():\n    lg=np.load(f); dm=dmarg(lg)\n    out=[np.sqrt((dm[(marg>=a)&(marg<b)]**2).mean()) for a,b in [(0,1),(1,2),(2,99)]]\n    # soft flip score: expected flips if noise were gaussian w/ item's |dm| under shrink k=1.25\n    from math import erf\n    print('%-6s dm rms by margin [<1, 1-2, >2]:'%n, np.round(out,3), ' overall %.3f'%np.sqrt((dm**2).mean()))\n# paired comparison v1 vs v3 per item on low margin\nd1=dmarg(np.load('lg_pass25.npy')); d3=dmarg(np.load('lg_pass29.npy'))\nm=marg<2\nprint('low-margin items: frac where |dm_v3|<|dm_v1|: %.3f'%(np.abs(d3[m])<np.abs(d1[m])).mean(), ' n',m.sum())\n"
}
```


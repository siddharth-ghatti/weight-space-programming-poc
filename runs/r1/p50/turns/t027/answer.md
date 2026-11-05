## reasoning summary

I'm checking the m=1536 results across all six layers: agreement is 97.7% and practice accuracy dropped from 92.5% to 90.4%. The disagreements seem to sit in low-margin examples, and the test set looks like it has many more of those. I'm estimating that roughly 30-40% of test examples might fall below a 0.85 margin, which would pull test agreement down to around 0.79 on those.

I'm checking whether the flips on test really cost accuracy, or whether the 8% disagreement overstates it. Low-margin cases are close to coin flips on correctness, so the accuracy drop should come in smaller than the disagreement rate. The logit-noise estimate of about 0.2 flip probability for those examples looks consistent with the observed 0.79 agreement.

Sticking with the 2-point accuracy drop as the real signal, the concern is whether the base's top choice is more often correct than its runner-up on those borderline cases. If it is, flipping them costs accuracy, and at 8% flips even a modest per-flip cost adds up to about a point. The in-sample numbers may understate this, since the base has seen these practice examples. A better test-time estimate would come from the test set's own margin distribution, which I don't have yet.

I'm weighing how much logit perturbation is acceptable, and whether a simulation on the base practice logits could tell me anything given that the in-sample calibration is off. A K-fold pseudo-test isn't possible without retraining. The expected accuracy change on test depends on how the top-1 vs top-2 correctness gap scales with margin for low-margin examples.

I'm working out the expected loss from flips on low-margin test examples, treating it as the density at zero margin times the top-1 versus top-2 correctness gap, integrated against the flip probability under margin noise with standard deviation around 0.25.

Working through the expected loss, a rough estimate puts it around a quarter of a point, since the margin density near zero on test is maybe 0.3 to 0.4 and the flip noise is modest. I also suspect the fine-tuned model is overconfident on test, which would shrink that loss further, so the risk looks small.

I'm estimating the random variance from flips alone: with around 24 flips on a 400-example test set, each roughly ±1, the accuracy change has a standard deviation near one point. That makes a sizeable drop on test a real possibility, so the flip rate on test needs to stay low, with the margin noise and density near zero being the key quantities to pin down.

I'm working out how much the flip rate has to drop: at a margin density near 0.35 and noise around 0.1, that's roughly 3% flips, about 12 on 400 test examples, which would keep the accuracy swing near 0.65 points. I'm also checking the pass threshold, since the base's 77.50% on 400 examples is exactly 310 correct, so 76.50% would mean losing about four examples.

Macro-F1 is the more volatile metric here, since the governance class is small (around 25 examples), so each governance flip moves the overall score noticeably. That means only a couple of adverse flips could eat the 0.02 tolerance, which makes the flip rate on test the thing to keep as low as possible, ideally under 2-3%.

I'm weighing whether to reduce the in-sample noise by choosing which layers to keep, since the allocation across layers is fixed by m but the selection isn't. I'm also considering whether better compensation from more probe data, which helped slightly at the early layers, could shrink the final-logit error, along with refitting the head to match the base logits.

I'm weighing whether a per-layer sensitivity test is worth the six extra passes, since pruning one layer at a time to m=1024 would show which layer drives the final logit noise. Layer 5's FFN looks well-compensable on its own, so the real problem is the shared m across all layers.

I'm weighing whether the remaining layers can absorb the work of a dropped one. Dropping layer 5 and fitting a linear map from its CLS output to the head gives about 0.949 agreement, which is worse than keeping all six layers at m=1536. Pruning FFN width still looks like the more efficient route.

I'm now asking whether including the layer input a in the regression, alongside the head's CLS features, would track the dropped FFN's output more closely than the CLS-only fit does. The noise at m=1024 is probably around 1.5 times the m=1536 level, so compensation quality matters more than it did.

I'm considering whether the FFN could carry a linear term in its input, since there's no direct skip path from the input to the output. One trick is to keep some hidden units in GELU's linear regime by giving them a large positive bias, so they pass a·w through unchanged and the output weights can realize a linear map.

I'm considering whether to drop a layer entirely and compare that against pruning FFN width across all six layers at similar parameter counts, since the noise per layer would show which option holds up better.

I'm comparing a 6-layer model at m=1024, which is about 23% smaller, against the m=1536 setup. I'm also wondering whether more probe data would improve the compensation, since the least-squares fit already cut the error at m=1024 by about 13%.

I'm going to estimate the margin noise standard deviation σ at m=1536 first, then use it to guess how many test examples would flip, assuming a test margin distribution with roughly 25-35% of examples below a margin of 1.

I'm checking the low-margin test-like subset next, where agreement is 0.79, and expect the flip count there to scale up. Then I'll compute the margin noise σ as the change in top-1 minus top-2 logit gap, using the base model's top-1 and top-2 indices.



At 1536 width on all six layers, the pruned model agrees with base 97.7% of the time and practice accuracy drops to 90.4%, with nearly all flips in the lowest-margin 10% of paragraphs. Since the test set will have more such cases and the macro-F1 margin is tight, I'm now quantifying the logit/margin noise and estimating the test flip rate.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'common.py').read())\ndef margin_noise(lg, BL=BL):\n    o = np.argsort(-BL,1); t1, t2 = o[:,0], o[:,1]; r = np.arange(len(BL))\n    mb = BL[r,t1]-BL[r,t2]; me = lg[r,t1]-lg[r,t2]\n    return (me-mb).std(), np.abs(me-mb).mean()\nfor nm, pth in [('drop4 orig', 'pass_3'), ('m1536 orig','pass_10')]:\n    lg = np.load(WORK/pth/'logits.npy'); print(nm, 'margin noise std/mean|.|', margin_noise(lg))\n# test-flip estimate: assume test margins ~ base practice margins rescaled down; base practice margin stats\no = np.sort(BL,1); mb = o[:,-1]-o[:,-2]\nprint('practice margin pctiles', np.percentile(mb,[5,10,25,50,75]))\nprint('logit scale std', BL.std(0))"
}
```


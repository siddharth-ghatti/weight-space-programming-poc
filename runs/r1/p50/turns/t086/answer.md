## reasoning summary

I'm checking the logit drift of the m1536 model against the earlier measurement, and it comes out smaller. Its accuracy and F1 are close to the base model's on the practice set, though the base still leads there. What I need to figure out is how the two models compare on test, where the base sits at 77.5 accuracy.

I'm estimating how many test predictions the edited model would flip, probably around 3 to 5 percent since low-margin examples are less stable. Working through the math, each flip costs roughly 0.3 points on average, which would put the test accuracy hit near 1.3 points. That's over the 1.0-point margin, which is a concern.

I'm weighing whether going smaller than the current 68.1M model is worth the risk, since the m1536 version already costs about 1.5 points on practice. Dropping a layer and shrinking the hidden width save roughly the same parameters, so the choice between them comes down to which keeps accuracy better. The real concern is how much test accuracy I'd lose, and whether that stays within the margin.

I'm trying to estimate test loss from the practice flips: 19 flips out of 1000, with base right on about 17 of them. Test margins are probably lower than practice margins, so there would be more flips there, but I don't have that margin distribution to work from. Given that uncertainty, I'm leaning toward keeping the model at high fidelity rather than shrinking it further.

I'm weighing whether to cut parameters further, but the risk of failing outweighs the gain, so safety comes first. I'm considering whether the noise at this size can be reduced, and one option is gathering more data for the compensation fit, which means running more probe passes per layer.

Now I'm checking whether the FFN inputs for several layers can be pulled from a single pass, since the hidden states I can read don't directly show the input to each layer's FFN without running attention, which I'm not allowed to do.

I'm working through whether the FFN input can be recovered from the layer's output alone. Since the output is a normalized sum of the FFN input and its transform, I'd have to invert that relation, and the unknown scale makes that look difficult.

I'm noting that for the last layer only the CLS token's output matters, since the classifier reads just that position, so its FFN only needs to be accurate there. I'm planning to use that to fit layer 5 specifically on the CLS token.

I'm looking at whether compensation for layer 5 can be targeted at just the CLS token, since only its logits feed the classifier. The fit would use roughly 2000 CLS samples, but with 768 output dimensions, that's not enough data to support many compensation units.

I'm weighing whether a smaller model could still hold accuracy, since m=1280 and m=1024 would save more parameters but the earlier numbers at m=1024 look too weak to trust. The sequential result there is much worse than the earlier non-sequential one, so I need to figure out which measurement to believe before going further.

I'm looking at how the m1536 result (.981) compares with m1024 (.896), a large gap with m1280 probably landing somewhere in between and not clearly safe. Given that m1536 already loses about 1.5 points on practice, I'm wondering whether it could still fail on test, so I want to estimate the test risk more carefully.

I'm now thinking about how to estimate test-time flips without test margins, so I'm looking at the margin profile of the practice examples the base gets wrong as a proxy for hard cases. I'm also considering modeling the logit noise directly: a flip happens when the noise on the top-two margin exceeds the margin itself, so I want the empirical distribution of that margin change on practice.

I'm treating the flip estimate as a margin-versus-noise question, so a flip needs the logit noise to exceed the margin. For test examples near the decision boundary, the base and the edited model are about equally likely to be right, so flips there should roughly cancel out in expectation.

I'm working out how much the edit would cost in expectation: the loss from a flip scales with the gap between the top two probabilities, so near-boundary flips add little, and only the in-sample results, where base was right on most low-margin examples, suggest otherwise.

I'm reasoning that the in-sample flips overstate the cost, since the base model has memorized those labels and fine-tuned models tend to be overconfident on unseen data, so each test flip should cost less. Still, with a 400-example test set, even a small expected loss carries real random variance, roughly ±0.8 points from a handful of flips.

I'm estimating the failure risk from the test flips: with about 10 flips, each leaning slightly toward loss, the chance of dropping more than 4 examples below the base's 77.5 comes out to roughly 21%. With only 5 flips, the chance of failing is considerably lower, around 5%.

I'm estimating the F1 risk now: the bar is 0.6896, and since governance is a small class, each governance error moves macro F1 noticeably, so the F1 check may be tighter than the accuracy one. The real unknown is how many test flips the m1536 model would produce, and I'm guessing somewhere around 12 to 20 given the lower test margins.

I'm checking whether the test set really is 400 examples, which I believe it is, and that makes the m1536 model look riskier than it first seemed. Next I want the margin distribution on practice, since the base's in-sample accuracy of 92.5% suggests it isn't heavily overfit, and that bears on how well practice margins stand in for test margins.

I'm listing ways to cut down the flips: more data for the compensation fits, choosing which units to keep more carefully, tuning the lambda, and a final end-to-end refit of the layer 5 FFN at the CLS position against the base outputs.

I'm considering using the final layer-5 FFN as a correction stage, fitting its output weights so the CLS vector matches the base model's final output, which would absorb errors that build up across the earlier layers.

I'm working out how to get the base model's layer-5 input at the CLS token, since the edited model's earlier layers would shift it. The idea is to target the normalized direction of the base's output, scaled by an estimated norm, so the compensation absorbs drift from layers 0 through 4. The base's unedited activations aren't directly reachable here, so I'm considering whether the edited model's inputs can stand in for them.

I'm checking whether layer 5's FFN can absorb the drift from earlier layers: if the base's a5 at the CLS token is available, I can compute its FFN output in closed form and fit layer 5's output weights to match it.

I'm now setting up the correction as a ridge least-squares fit on the layer-5 output weights, mapping the edited model's CLS input plus its kept-unit activations to the base model's target, with the regularization strength picked by cross-validation. Since there are more kept units than samples, I'm also weighing the errors by how much they shift the logits, so the fit focuses on the directions that matter for the classifier.

I'm reasoning that the logit error depends only on a few projected directions, so I could fit just those 4 to 8 directions rather than all 768. Each still has m features and only about 1000 samples, so ridge regularization would be needed to avoid overfitting. I'm also considering whether layer 5's attention could be adjusted to correct the CLS output, though that looks more complicated.

I'm weighing whether refitting the classifier head on the final CLS would help, but my notes show it didn't improve things (0.981–0.983), suggesting the errors aren't linearly recoverable there. The layer-5 FFN correction has more nonlinear features, though it probably runs into the same limit. Next I'm checking how many tokens the probe passes have actually yielded per layer, to see whether the data is enough to reduce the layer 0–4 noise.

I'm considering a simpler alternative: letting the kept units carry a linear correction in the input, but a full linear map from the 768-dimensional input can't fit into the FFN's structure without the residual path, which the layer norm blocks. That leaves the kept-unit correction as the main tool.

I'm weighing whether the gelu units with large positive pre-activations, which act almost linearly, could be handled by simple linear functions of the other kept units. The compensation fit already does that. Greedy OBS unit selection didn't beat variance ranking on holdout, so the realistic gains from here are limited. I'm now deciding between keeping m1536 and moving up to m2048, which looks only slightly better, though it's an older non-sequential version.

I'm weighing whether the m1024 option is worth it, since it trims parameters further, against the risk of losing too much accuracy. I'd like a test-like proxy for the margins, but the practice logits are in-sample, so I'm considering whether the margin difference between the edited and base model can stand in for test behavior.

I'm going to estimate test flips from the practice data by comparing the margin-change distribution against the practice margins, and I'll treat the base model's 75 wrong practice items as a possible stand-in for the hard part of test. I'll scale the practice margins down, as a pessimistic case, and see how many flips that gives.



The 6-layer m=1536 model matches base on 98.1% of practice items, with accuracy .910 against base's .925. I'm now estimating test risk by comparing the logit noise with how close base's predictions sit to the decision boundary.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nlg=np.load('lg_pass25.npy'); lg24=None\ns=np.sort(BL,1); marg=s[:,-1]-s[:,-2]\ntop=BL.argmax(1); \n# change in margin between base top and each other class\nd = (lg - lg[np.arange(1000),top][:,None]) - (BL - BL[np.arange(1000),top][:,None])\ndn = np.delete(d.reshape(-1), np.arange(1000)*4+top)  # noise on pairwise margins\nprint('pairwise margin noise std', dn.std(), 'p95 |.|', np.percentile(np.abs(dn),[50,90,95,99]))\nprint('base margin percentiles', np.percentile(marg,[1,5,10,25,50]))\nprint('frac base margin <0.5', (marg<0.5).mean(), '<1', (marg<1).mean(), '<2',(marg<2).mean())\nwrong = top!=y\nprint('base wrong margins pct', np.percentile(marg[wrong],[10,25,50,75,90]))\n# flip prob given margin bin\nfor lo,hi in [(0,.25),(.25,.5),(.5,1),(1,2),(2,99)]:\n    sel=(marg>=lo)&(marg<hi); print(lo,hi,sel.sum(), (lg.argmax(1)!=top)[sel].mean())\n"
}
```


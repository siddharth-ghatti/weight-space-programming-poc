## reasoning summary

The in-fold ridge scores are overfit, with near-perfect training scores that make the kernel overweight them and then do poorly on test, so I need nested out-of-fold scores instead. The M0 benefit also seems to need many dimensions, since 64 PCA dims only capture part of it, and that isn't cheap. Given the 97.8% threshold and the noise, I'm going to step back and set a realistic target, starting with a range of 0 to 8.

I'm weighing whether a smaller pruned model can keep accuracy: the 1024-feature kernel sits about 0.7 points below base, which probably fails the 97.8% bar. So I'm looking at keeping more layers or more features, and at improving the pruning itself by correcting drift so the pruned hidden states stay closer to the base ones.

I'm weighing whether unit selection or drift correction can get the pruned FFN closer to the base hidden states, since the m GELU units only span part of what's needed. The attention output weights would be hard to refit without the context vectors, so I'm considering whether to reduce what has to be preserved, perhaps by focusing only on the pooled features at the end.

I'm weighing whether greedy forward selection of the FFN units could cut the error, and I'm planning to test it offline on layer 4, though the FFN inputs I'd need for a held-out check aren't available from the cached data.

I'm considering whether the attention sublayer could be recomputed in numpy from the attention probabilities and token hidden states, which would avoid having to expose every layer at once. The FFN side already works that way through the existing activation helper, so fitting a closed-form correction for the attention output might be reasonable.

I'm ruling out running the model in the sandbox to get around the pass budget, since reconstructing attention outputs in numpy just to stand in for exposure passes would defeat the constraint. That leaves the FFN fit limited to the 64 exposure sentences per pass. My plan is to redo the sequential pruning with drift correction at m=1024, which costs about 10 passes, then two more passes to read the features for evaluation.

I'm planning a base pass to read the token hidden states for the exposure sentences at every layer, so I can measure how much drift the pruned network accumulates before deciding whether drift correction is worth doing. Without those base activations there's no way to test this offline.

I'm checking the drift in the pruned network layer by layer, using base hidden states for the exposure sentences, so I can see whether drift correction is worth the passes. The plan is to refit each FFN against the base targets, starting from the current upstream activations.

I'm weighing whether drift correction is worth doing by working through the error budget for a layer whose input has already drifted from the base input. If the pruned FFN just reproduces its own drifted input, the output error is the upstream drift plus the change in the FFN's response to that drift, which is the quantity I'd need to measure.

Now I'm comparing the drift-corrected error against plain replication, with the same residual term tracked across both. The unit features only capture a slice of the upstream drift, so I'd expect something like a 10-30% reduction, and the fit from only 64 sentences may not generalize well.

I'm weighing whether per-layer variation could help, but the constraint is that every kept layer keeps the same number of FFN units, so that's off the table. Dropping whole layers saves more per layer than trimming FFNs, since attention is a large fixed cost at m=1024, though earlier results suggest removing layers 4 and 6 already costs around half a point.

I'm considering a non-sequential approach instead: fit the pruning for every layer at once on the existing exposure sentences, accepting stale drift, and then read features for evaluation in one or two passes. That would let me compare different intermediate sizes like 1536 or 2048 without paying for nine sequential passes each time.

I'm planning a cheap check of the accuracy-versus-size curve: fit all layers at once on the existing exposure sentences for m=1536 and m=2048, then read the 1024 mean features for each in a single pass. Since 0.3-point differences are close to noise, I'll also consider reading all 1496 features in two passes for a more reliable comparison.

I'm thinking about whether to judge the candidate configurations by a smoother signal than CV accuracy. Accuracy differences of about 0.3 points on 1024 samples are within noise, so I'm considering comparing the mean true-class margin or out-of-fold error instead, which should separate the variants more reliably.

I'm thinking about whether the full pipeline can reach the target accuracy, since the base M9 features alone only give about 98.13 on the full set. That suggests I'll need richer pooled features from the final layer, such as attention-weighted pools over different token subsets alongside the uniform mean.

I'm now weighing whether a carry head in layer 8 could supply extra pooled signal, since the final layer only sees h9. I can't check any of these pooling options against token-level data yet, because the base activations exist only for 64 sentences.

I'm weighing whether the CLS position could carry a pooled summary from layer 8, since the pool layer's CLS query reads every token's h9 and the other tokens only see h8 through attention. Changing CLS's state there might not interfere with the other tokens, but I can't verify that offline.

I'm working out whether a head in layer 8 could be repurposed to pool for the CLS token without disturbing the other positions. Changing that head would alter every token's output there, so it would cost the function it was already serving. Earlier carry heads in layers 2, 6, and 10 already gave up one head each for this kind of signal.

Now I'm weighing whether the pool layer's extra heads can carry signal beyond the uniform mean, though I can't tell which patterns would help without token-level data. The base M9 result of about 98.13 is marginal, so I'll need richer pooled features, likely by carrying early-layer information forward, and I'm considering how reserved dimensions might hold it.

I'm looking at whether the reserved dimensions could carry the M0 and M1 information forward, since the CLS position is attended by every token and so can't hold it cleanly. The current 8 reserved dims already hold the M2 and M6 scores, and they're cut from most layers, so widening that channel is the open question.

I'm now asking whether the reserved dimensions could be repurposed for early-layer information, since the carry head at layer 2 only sees h2 and so can't supply M0-style signal directly. Eight dimensions is a narrow channel, so I'm wondering if pushing more pooled context into the CLS position, or a second pooled readout alongside M9, would close the gap that M1 plus M9 suggests.

Now I'm looking at whether cheap pooled statistics from layer 9, like mean squared activations or norms, could add signal. Since the pool layer has 12 heads, each could attend differently, and with a fixed query the attention reduces to softmax-weighted pooling by a per-token score, which might capture class-evidence more directly than the uniform mean.

Next I'm considering whether a pooled readout over the layer-9 token states could add signal, since the pool layer's heads could weight tokens differently than a uniform mean. Evaluating those designs means reading the pool output for all 1496 sentences, which takes a couple of passes per design, so I'm planning a small number of candidates within my remaining budget.

I'm planning a sequential pruning pass at m=1536 and possibly 2048, reading 1024 mean features from M9 for each, and comparing them with the base and m=1024 through KRR cross-validation. Once I pick a size, I'd redo the sequential fit with drift correction, and then build and check the pool layer against all 1496 sentences.

I'm checking how the exposures are used: the layer-0 fit is exact since its input is untouched, while the later layers were fit against exposures from the old pruned upstream, so I'm weighing whether a drift correction toward the base hidden states would over-correct once the upstream is updated.

I'm weighing whether a drift correction toward the base hidden states would over-correct, since the upstream has already been updated. I can test that cleanly in a single pass, though a blended target might be safer. Scoring the candidates by parameter count, the m=1024 variant comes to about 63.8M, roughly a 41.7% reduction.

I'm weighing a larger m=1536 variant without a drift correction, expecting it to land partway between the base and m=1024 results. The m=768 option at about 59.9M is the next thing to set against that. I'll first check m=1536 by refitting from the existing exposures and reading the M features for 1024 ids in one pass. Then I'll compare a second run that reads base token hidden states for the exposure ids, so I can test a drift-corrected fit.

I'm weighing whether to refit the carry heads with drift correction against the stale exposures, since that's a two-pass check. I'm also considering whether the classifier itself could do better by combining the pruned features with the carried M2 and M6 scores from the original model, though those in-sample numbers are inflated and the honest out-of-sample estimate is closer to 98.2.

I'm considering whether the layer 2 and 6 carry heads could be rewritten to output different scores rather than the summed reserved dimensions, which loses information. I'm also weighing a pooler with 768 tanh anchors and a pool-layer FFN as the nonlinear classifier, since two nonlinear stages might capture more than the current setup.

I'm checking where the final pipeline stands with the current pruned features, since the pruned M9 score is 96.63 against the base 97.31 on the 1024 subset. Keeping one more content layer would cost more parameters and wouldn't obviously help, so I'll try to improve the pruned network instead, starting with a base token read for the exposure ids to enable drift-corrected fitting.

I'm working out the drift-corrected target: for each token, I'd rescale the base next-layer hidden state into the pruned network's pre-norm range, using that token's own mean and spread, then subtract the drifted input to get the residual the feed-forward block should learn.

I'm settling on the fit: the pruned feed-forward weights get a ridge pull toward the original output weights of the kept units, so the solution stays close to plain pruning and the mixed target only nudges it. Next I need a base token read for the exposure ids, with layers 1 through 10 included to cover the h_{l+1} targets.



Linear scores and PCA didn't recover M0's benefit, and refitting for replication alone is limited, so I'm trying drift-corrected fitting against the base model's hidden states, which means reading those for the 64 exposure sentences next.



## tool_use: run_python

```json
{
 "code": "import json\ne=json.load(open('expo_ids.json')); print(len(e)); print(json.dumps(e))"
}
```


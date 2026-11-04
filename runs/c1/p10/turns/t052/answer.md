## reasoning summary

I'm noticing the carry9 score of 99.96% is leakage, since the carried scores were fit in-sample on all practice sentences, so they can't be used for cross-validation. The notes already said to exclude those reserved dimensions. Still, those scores will also show up in the edited model at test time, and whether they help there is something I can't evaluate yet. The original editor's model did use them and reached 98.31 on test.

Excluding the leaked scores is the honest choice for evaluation, but the gains from adding linear layers are small, around +0.1 to +0.3, while pruning would cost about 0.5, so that trade isn't worth it. The notes show only the depth-9 model clears 98 in CV, so reaching a smaller model with fewer layers looks hard, and combining M0 with M8 doesn't work because the pooling stage would need all 768 dimensions of M0.

I'm weighing whether to go with the m=3072 design: it lands around the original's 98.31 on test, and the smaller parameter count might be worth the risk, though a failure would cost everything.

Now I'm weighing the m=2560 option against the m=3072 design, with expected test scores around 97.9–98.1 and a failure risk of roughly 30–40%. I'm also thinking about whether the pruning itself can be improved cheaply. The layer-0 mean shift of about 11% suggests the Gaussian proxy's means are off, so I'm considering measuring the actual shift between base and edited hidden states at the layer-1 output and using that to correct the compensation.

I'm weighing whether shifting the layer-0 output bias by the residual mean would reduce the drift, though deeper layers would still carry their own shifts. I'm also considering whether selecting units by magnitude, weight-row norm times weight-column norm, might beat the Gaussian importance estimate, but I can't tell yet which of the two will hold up.

I'm weighing a smaller pruning step, m=2816, which would drop about 16.5% of parameters with an estimated damage of roughly 0.2. I'm also considering whether layer 8 could handle the pooling itself for the CLS token, by repurposing some heads to attend uniformly.

I'm weighing a last-layer trick: since only the CLS output matters there, every head in layer 8 could be repurposed to attend uniformly, and the FFN could become the kernel machine. That would give a pooling-plus-kernel layer built on M8 features, but those reach only about 97.8, which isn't enough on its own.

Now I'm weighing whether the depth-9 result matters for the m choice: M9 at 98.66 is only slightly above M10 and M11, so the extra depth isn't clearly needed, and a 10-layer design is the minimum. That leaves the width m as the only real lever, with 2560 saving about 8M parameters over 3072 but carrying the higher failure risk.

I'm checking whether choosing units by a different criterion could help, since the drift at layer 0 is already about 14% even though the bottom units should barely matter. That points either to the Gaussian importance proxy being inaccurate or to special tokens with outsized FFN outputs skewing the means.

Now I'm weighing whether to pick the units by a different criterion, since the drift at layer 0 is already large. I'm also checking how much this matters in practice: the leave-one-out score fell from 97.75 to 97.27 after pruning, so that is a real information loss even with the head refit on the edited features.

I'm planning to check the per-layer drop in the subset run, comparing leave-one-out scores for base and edited models at each depth, to see whether the loss starts early and carries through. I'm also weighing whether a small accuracy margin from another source could make a pruned model safer, though the linear add at layer 0 would need a head that disturbs the layer.

I'm weighing whether the carried reserved-dimension scores can be used at all, since fitting a head on practice data would make them look perfect in-sample and overweight them. Any head I refit around them would need a clean holdout to judge whether the carry actually helps.

I'm sketching a joint fit: one kernel ridge model with the RBF kernel on the final features plus a scaled linear term on the layer-2 averages, trained on all practice data together so nothing is stacked or leaked. The linear weights would then be written into the layer-2 carry head, but the per-token LayerNorm scalings in layers 2 through 8 make that write-in delicate, and I'm not yet sure how the original editor compensated for them.

I'm weighing whether a pilot run at m=2816 would settle the choice between it and m=3072, since it would drop about 16.5% of parameters. I'd read just the layers that matter, 9 and 10, on 1024 ids before reading the remaining 472, and judge the final call from those numbers.

I'm reasoning about whether the m=3072 design needs a fresh read of the data, or whether the final CLS can be derived from the layer-8 output directly. Computing that pooling in numpy would mean reimplementing part of the model, so reading the outputs is the safer path.

I'm settling on a plan for the m=2816 experiment: prune the layer-0 structure with the same proxy selection, then read only layers 9 and 10 on 1024 ids. If it holds near the base score of 97.75, I'll read the remaining 472 ids, fit the head, and verify, with one spare pass kept for a fallback.

I'm now comparing the m=3072 route against a pilot at m=2816. Choosing m=3072 directly means two reads plus one verification pass, which leaves two spare passes, so one experiment is affordable. I'm also considering whether a better pruning selection could help m=2816, though I can't attribute drift to individual units with what I have.

I'm checking whether the pruning proxy is the problem, since 14% drift at layer 0 seems large if the bottom units should matter little. I want to see how concentrated the importance scores are, because if the bottom 512 units carry only a small share of the total, that drift points to a poor selection criterion rather than an inherent cost.

I'm wondering whether the proxy-based importance scores are the source of the 14% drift at layer 0, or whether I'm misjudging the distribution. The notes assumed a Gaussian input, but I'm not sure which hidden states that assumption was fit to. A data-based check would need token hidden states at every layer from one read of 64 ids, though I can't invert the layer outputs back to the FFN inputs without recomputing attention.

I'm weighing whether to run the m=2816 pilot anyway, since the Gaussian proxy is the more defensible route and reimplementing the forward pass on approximate inputs isn't. The expected gap is around 0.2, which is within noise on this subset, so the pilot would probably leave the choice ambiguous.

Comparing the expected test accuracy across the candidate designs: the m=3072 model should land near 98.5 once a realizable head is used, leaving about 0.7 of margin over 97.8. The m=2560 option sits closer to 98.0, which is too thin a margin, while m=2816 might reach around 98.3.

Now I'm working out how big the test set must be, since the stated accuracy of 98.31% only fits certain sample counts. A set of 500 would give a standard error near 0.58%, which is large, while around 1500 would tighten it to about 0.33%. I'm checking which whole-number denominators are consistent with that exact percentage.

I'm estimating how large the test set must be, since the 98.31% figure only fits certain whole-number sizes around 500 to 1200. At a few hundred examples the standard error is around half a point, so the m=2560 option is quite risky while m=3072 still looks reasonably safe. The original model's score on that same set is the fair comparison, since both are measured on identical examples.

I'm leaning toward m=3072 as the safer pick, since m=2816 looks marginal and the parameter gain doesn't justify the risk of a catastrophic miss. I'm also wondering whether the pooling could be folded into layer 8 itself, so the final layers' token processing stays intact.

I'm checking whether the final CLS state could be pooled inside layer 8 itself, but the mean over tokens only exists after attention, so it can't be folded into that layer's FFN. Relying on layer 8's CLS output alone loses too much, since its leave-one-out score of 97.8 is below the margin I need.

Considering whether layer 8's pooled features could be mixed with its normal CLS processing. Mixing the mean into the attention would give something like C9 plus M8, but I doubt it would beat M8 alone, which scored about 97.8. The mean over all tokens only exists after layer 8's feed-forward pass, so that information is out of reach inside layer 8.

I'm comparing the layer-8 and layer-9 scores: a 0.9 gap on 13 samples, while the 3-seed CV showed only about 0.45. The direction holds in both, but the size of the gap is uncertain, and I'm wondering whether the fixed hyperparameters, tuned on layer 9, favor it.

I'm considering whether the removed units' contribution could be approximated by a linear term that the kept units absorb, though that doesn't seem workable since GELU units aren't mostly linear. A better angle might be choosing which units to prune by their joint variance under the Gaussian proxy, since correlated units can cancel each other's contribution.

I'm weighing whether to go with m=3072 despite its modest 13% reduction. Before deciding, I want to estimate how much accuracy m=2816 would lose from pruning only 256 units per layer, by comparing the squared importance mass of the bottom 256 against the bottom 512 in each layer.



The carried reserved-dim scores are leaked (99.96% CV from in-sample fitting), so they stay excluded, and the layer 0..8 architecture stays as it is. Next I'll check whether the bottom 256 removed units carry much less energy than the bottom 512 to decide if a milder prune at m=2816 is worth one experiment.



## tool_use: run_python

```json
{
 "code": "p=np.load('ffn_proxy.npz')\nfor L in range(9):\n    imp=np.sort(p[f'imp{L}']); tot=(imp**2).sum()\n    print(L, 'bottom256 %.3f  bottom512 %.3f  bottom1024 %.3f' % ((imp[:256]**2).sum()/tot, (imp[:512]**2).sum()/tot, (imp[:1024]**2).sum()/tot), 'imp range', imp[0].round(3), np.median(imp).round(3), imp[-1].round(3))\n"
}
```


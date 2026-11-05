## reasoning summary

No change came from the holdout tweaks in L3 and L4, so the end-to-end logit noise doesn't seem sensitive to those local edits. It may be driven by L0–L2 or by something else entirely. I'm planning a few single-pass experiments that prune only subsets of layers to see which ones dominate the end noise, and I'll need to sort out the uniform-width constraint on m as I set them up.

I'm weighing whether zeroing out the dropped W2 columns in pruned layers reproduces slicing exactly, so I can run diagnostics with a uniform m=3072. The plan is two passes: one pruning L0–2 at 1536 with L3–5 kept full, and one pruning only L3–4. Whether these are worth running depends on what I'd do with the result, given that local fixes already look saturated.

I'm going to check how the logit noise scales with m, using the per-layer error figures from the earlier holdout runs. If the noise grows roughly with the square root of the error energy, the rough estimates put m1280 near 22 flips and m2048 near 13, with m1792 falling in between at about 16.

I'm working out how the flip counts scale with pruning width. The error-energy ratios from the earlier runs suggest noise grows somewhat faster than linear propagation would predict, so I'm estimating roughly 26 flips at m1280, about 14 at m1792, and about 10 at m2048.

I'm now working through the rough failure risk for each pruning width, using a crude model that scales test flips from the observed practice flip rate. That puts m1024 around 37%, m1280 near 27%, m1536 near 18%, m1792 near 12%, and m2048 near 8%, with full width at zero. These numbers lean heavily on my assumed multipliers, so I'm holding them loosely.

I'm estimating how much the test-set flip count could differ from practice, since the test inputs are unseen by both the compensation fit and the base model's training. The main uncertainty is how many test flips actually cause losses versus gains, and I'm treating that split as roughly 55:30 rather than the 17:2 seen in practice.

I'm weighing the scoring rule here: if it's just parameter reduction conditional on passing, m1536 and the more aggressive widths come out roughly even under my risk estimates, so the choice hinges on how much I trust the failure probabilities.

Under linear scoring the choice barely matters from m1024 to m1536, but lower widths carry more variance, so a higher width inside that flat region looks like the safer pick. The real lever is still reducing noise at a fixed width, which makes me want to find out where the noise comes from. I'll run two diagnostics with zeroing-simulated pruning: one pruning L0–L2 at 1536 with L3–L5 full, and one pruning L3–L5 with L0–L2 full.

I'm going to back up the current edits first, then restore the original W2 and bias for the full-width layers so the two diagnostic passes can be set up cleanly.

I'm weighing whether to spend a couple of passes mapping the m1280 and m1792 agreement points, since that decision matters more than local fixes. I'm also wondering whether getting more calibration data for the early layers would pay off, though the cost per layer is the same either way. The harder problem is the token-level idea: fitting each layer on paired base and edited activations would require tracking both sides together.

I'm now wondering whether the token-level errors could be partly cancelled by fitting each layer to the residual errors left by earlier layers at the same token, which would need paired base and edited activations. Since that didn't help at the CLS level and would cost several more passes to collect the paired data, the gain looks uncertain.

I'm weighing whether to check how the base model's margins on atypical practice items compare to test, using a cross-validated ridge probe on base CLS features to estimate the test margin distribution more directly than my crude multipliers.

I'm estimating test flips at m1536 at roughly 11 to 19, based on a test low-margin density of 1.5 to 2.5 times the practice rate. That's worse than my earlier figure, so I need to reconsider how the per-flip outcome splits on the test set.

I'm estimating the test-set risk per flip: assuming the base model is right about 45% of the time on low-margin items and the second choice about 35%, each flip nets a small expected loss with high variance. Working that through, 11 flips give roughly a 13% chance of a loss of five or more, and 19 flips about 25%. Next I'll run the same calculation for the m2048 flip count, which is around 6 to 11 on test.

I'm working through the failure-risk numbers for each width: m1536 looks like roughly 13 to 25 percent, m2048 about 4 to 13, and m1792 around 8 to 18, so every option carries meaningful risk. Getting the chance under 5 percent would need six or fewer test flips. I'm also wondering whether the noise comes from a handful of fragile items, since the practice flips all sit at low margins, and whether trimming a bit less width is worth it given the small savings.

I'm checking whether the heavy-tailed logit changes cluster in particular sequence lengths, since longer inputs might accumulate more error, so I'll compare the logit differences against length.



v2 matches v1 almost exactly (98.0% agreement, same logit error), so the local per-layer gains aren't reaching the final logits. Before choosing m, I'm checking whether the logit noise clusters by input type, such as long versus short paragraphs.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nlens=json.load(open('lens.json')); print(type(lens), (list(lens.items())[:2] if isinstance(lens,dict) else lens[:5]))\n"
}
```


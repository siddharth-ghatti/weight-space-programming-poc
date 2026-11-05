## reasoning summary

I'm checking how the practice set's margins compare with the test set. About 5.5% of practice examples fall under a 0.5 margin, and the base model's wrong predictions there have a median margin of 1.1, so it's confidently wrong on many. The test set's base accuracy is 77.5%, so some of its errors look like label noise or ambiguity. I'm estimating the test set has roughly twice the low-margin density, around 11% under 0.5 and 24% under 1, and working out expected flips per margin bin, starting with P=.46 for the 0–0.25 bin.

I'm estimating the test flips at about 15 out of 400, and with the base model right on roughly half of those low-margin cases, the expected loss per flip comes out near 0.25. That puts the chance of losing five or more items somewhere around 40%, which is a worrying number, though I'm not yet sure my guessed probabilities are right.

I'm working out how often the test flips would cost us: at roughly 15 flips with a per-flip expected loss near 0.25, the odds of losing five or more items look high, around 40%. If I cut the flips to about five, that risk drops to a few percent. My probability guesses for the test margins are rough, so I'd check the per-flip loss before trusting these numbers.

I'm reconsidering the expected loss per flip: at a margin near zero between two classes, a calibrated model gives roughly equal probabilities, so the loss is close to zero, and at a 0.25 logit margin it comes to about 0.1. Since most flips sit at small margins, I'm estimating a mean loss of about 0.1 per flip with variance near 0.8, and I'm starting to compute the total for 15 flips.

I'm weighing the test flip count against the risk: at about 8 flips the chance of losing five or more items is around 7%, and at 4 flips it drops to about 1%, so the number of test flips is what matters most. I'm estimating the test flip rate at 2–4%, which would mean roughly 8–16 flips.

I'm weighing a moderate size increase against the lower failure risk it buys, since even the middle option carries some chance of too many flips on the test set.

I'm weighing whether to drop a whole layer and keep the rest at full width, since the FFN width can't be made non-uniform without going back to full size.

I'm considering whether dropping a layer and widening the rest could beat six layers at m1536, but the arithmetic puts it at roughly the same size with worse fidelity, so I'll stick with six layers. The open question is how to improve fidelity at a fixed width, and I'm starting to look at how the per-layer local fit's error gets amplified once it passes through the LayerNorm after the FFN.

I'm weighing whether weighting the least-squares fit by each output dimension's LayerNorm gain would help, but since each output dimension is solved independently, that weighting wouldn't change the solution unless some shared constraint ties them together.

I'm noting that the per-output-dimension least-squares solve doesn't interact across dimensions, so weighting by LayerNorm gain or token role changes little unless I tie things together. The bigger lever seems to be compensation with more data, since train error is noticeably below test error, so I'm about to check how many tokens each layer's FFN input has.

I'm checking whether the dropped-unit error is actually measured as a relative residual, and I'm considering choosing the regularization strength per layer using holdout data. I'm also thinking about merging similar units rather than dropping them outright, starting with clustering the units by similarity.

I'm considering merging units whose input weights are nearly parallel rather than dropping them, though gelu isn't homogeneous, so that only holds approximately. The stronger idea is that units firing in the linear regime just compute a linear map of the input, which could be captured by fewer units.

I'm considering a way to avoid dropping units entirely: fold the nearly linear ones into a smaller set of units with large positive biases, so their combined effect is a linear map of the input. The approximation holds only while the pre-activations stay well inside the linear region of GELU, so I'm checking the scaling needed to keep that true in float32.

I'm considering whether a linear term in the compensation could capture much of the dropped-unit contribution, since any linear map of the input can be built from extra units with large positive biases. That would mean adding a full linear component on top of the kept units, though it could cost up to 768 units to represent in full.

I'm weighing a joint fit where some of the kept units are replaced by linear terms, since a unit in the near-linear regime of GELU can carry a linear map of the input. I'll compare that against plain units with least-squares compensation on held-out data, using a reduced-rank fit on the linear part.

I'm going to test a linear term on the existing data, fitting it alongside the kept units and then truncating it to low rank, since a near-linear unit can carry a linear map of the input. I'll check it with holdout splits grouped by sequence, and I'm leaving layer 5 aside because it only has CLS tokens.



Practice flips sit mostly at base margins below 0.5, so I expect 8–16 flips on the test set and a 10–20% chance that m=1536 fails. Next I'll check the data sizes and builder helpers before testing a linear-map idea offline.



## tool_use: run_python

```json
{
 "code": "print(open('builder.py').read()[:3000])\nfor L in range(6):\n    for f in [f'ffn_in_L{L}.npy', f'ffn_ed_L{L}_B1.npy', f'ffn_ed_L{L}_B2.npy', f'ffn_in_L{L}_cls.npy', f'ffn_ed_L{L}_cls.npy']:\n        if os.path.exists(f): print(f, np.load(f, mmap_mode='r').shape)\n"
}
```


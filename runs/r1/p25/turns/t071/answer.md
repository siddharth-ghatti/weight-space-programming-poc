## reasoning summary

Adding more data gives modest gains, around 11 to 17 percent lower error, which likely reflects that the long-paragraph tokens are distributed differently from the short test paragraphs. That works out to only about a 5 to 9 percent cut in hidden perturbation, so it's not a major improvement. I'm weighing whether that's worth the cost.

I'm wondering if there's a bigger gain available than just adding more passes. Even with unlimited data, the error at m=2048 for L0 looks like it can't drop below about 0.021 to 0.035, so I'm checking whether changing the kept units' input weights could get past that limit.

I'm sticking with the idea that the per-layer error at a given width is close to a floor, so the real lever is choosing how many units to keep. Keeping 2048 units gets roughly 0.985 agreement, and more data might push that a little higher. Trading that against dropping an entire layer looks worse, since refitting the head on the truncated representation only reached about 0.955.

I'm weighing whether a layer-merging trick could avoid dropping a layer outright, but it doesn't really fit: the FFN runs per token, so it can't reproduce the attention that layer 5 uses to pull information into CLS. That leaves FFN pruning with six layers kept, and the real choice is the width somewhere between 1536 and 2048.

I'm considering whether to accept m=2048 with more data for the later layers, or drop to m=1792 and get roughly the same fidelity. Weighting token rows by downstream attention is an option, but it's messy to do without gradients. I'm going to compute the cross-validated error for L0 at m=1792 to compare against the older m=2048 numbers.

I'm weighing whether m=1792 with the extra data would match the current m=2048 fidelity, which would save about 3% of the parameters. Then I still have to decide whether the current agreement level is acceptable at all, and my rough risk estimate for m=2048 is too crude to settle it.

I'm going to get two more passes of data for layers 1 and 2, then rebuild both the m=2048 and m=1792 candidates to compare them directly.

I'm considering whether the first layers could benefit from a sequential correction, since their input error propagates from layer 0's FFN through attention. A corrected target for layer 1 might absorb some of that propagated error that sits in the residual path.

I'm considering whether a sequential correction would help the lower layers, since their upstream error passes through attention. The catch is that it costs extra passes to measure. Given that the layer-5 correction didn't help, I'm doubtful, so I'll skip it and put the passes into more data for layers 1 and 2, the hardest ones.



More data helps only moderately: layer 0 error falls 11% at m=1536 and 17% at m=2048, so m is the real lever. I'm running the layer-1 probe pass (pass 15) on the same 32 paragraphs next.



## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0647",
  "p0664",
  "p0697",
  "p0537",
  "p0533",
  "p0679",
  "p0762",
  "p0513",
  "p0694",
  "p0132",
  "p0494",
  "p0573",
  "p0563",
  "p0597",
  "p0463",
  "p0388",
  "p0572",
  "p0977",
  "p0249",
  "p0318",
  "p0465",
  "p0552",
  "p0427",
  "p0601",
  "p0464",
  "p0596",
  "p0447",
  "p0952",
  "p0781",
  "p0230",
  "p0052",
  "p0591"
 ],
 "layers": [
  2
 ],
 "outputs": [
  "token_hidden"
 ]
}
```


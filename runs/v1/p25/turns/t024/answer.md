## reasoning summary

I'm going to sanity-check the pooler and layer-norm algebra before committing to a full pass. The pooled term for each class combines the kept part of B_k·x with the channel part divided by c, where x is the normalized value (v − μ)/σ and the channel shift depends on A_k, M, b_k and the hm_k/c correction.

I'm checking the pooled-argument algebra: the layer-norm division by c and the B_k·h_keep term combine to a sum that's small after dividing by σ, which I estimate around 5. That puts the pooled arguments in a range where tanh is well-behaved, though the differences between classes are small and should still survive float32. I also want to confirm the final LN2 is effectively an identity, then run a quick predict pass on a sample to check accuracy.

Now I'm weighing whether the offline logits can be checked against the model's outputs, using a monotonic tanh mapping and comparing argmax agreement and correlation on a sample.



Edits are saved for the layer-11 readout, pooler and classifier, with 100% offline training accuracy as expected. I'm running one predict pass on the edited model to confirm its argmax and ordering match my offline logits.



## tool_use: forward_pass

```json
{
 "kind": "predict",
 "model": "edited",
 "n": 300,
 "seed": 1
}
```


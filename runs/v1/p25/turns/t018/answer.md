## reasoning summary

The linear decoding from h11 back to h0 only reaches an R² of 0.38, so token identity isn't really linearly recoverable from the deep layer. Linear token decoding is off the table. I'm now looking at a different readout: pooling at a shallow layer and then running the remaining layers as a readout network.

I'm weighing whether CLS-carry could work, by measuring how much attention the CLS token gets from the others. Its own layer processing might dilute whatever information it carries, and I'm estimating how much that dilution costs per layer.

I'm considering a variant where the CLS token gets gated off in later FFN layers, so its carried information isn't overwritten. The idea is to add a negative bias along a marker direction to the FFN input weights, so the CLS state's FFN output collapses to just the bias while normal tokens are left mostly alone. I still need to find a marker direction where ordinary tokens have near-zero projection.

I'm working out the numbers for a marker-direction gate: a random unit vector gives token projections with std around 0.8, so the perturbation to ordinary tokens would be on the order of β times that, while the CLS shift is roughly β times M. Trying β = 1 pushes too much noise onto the other tokens, so I'm checking whether β = 0.25 keeps CLS switched off while leaving the rest mostly intact.

I'm now looking at whether the CLS token's attention output could be made constant, either by having it attend only to itself or by using uniform attention, and weighing whether the sentence-dependent value mixing might still be useful if it lives in a separate subspace. I'm also considering the value-side route, where the CLS attention output is set through the output projection.

I'm considering making the CLS token attend only to itself, by adding a marker-direction term to the query and key projections so its self-attention dominates. The difficulty is that this same term would also shift how other tokens attend to CLS, unless the chosen direction is nearly orthogonal to their queries.

I'm weighing a cheaper route instead: keep mean pooling at layer 11 and add a sharp, keyword-seeking attention head on top, with query vectors fit from token-level class evidence. That could add a point or so without the risky CLS gating work.

I'm setting the attention so the head's query is a constant β·w_c, with the key row picking out the token's projection onto the class direction. That makes the softmax weight each token by its class evidence, which gives a sharp, keyword-seeking pooling for that head.

I'm weighing a second route: pooling at layer 0 into the CLS token and having its later-layer processing carry that information forward. Testing this would mean reading token states across all 1496 sentences, which is too many passes to be worth it right now.

I'm thinking about the cleanest way to keep the CLS state's own information separate from the pooled deep features at layer 11, so the readout doesn't blur them together.

I'm weighing whether to pool the deep token states into the CLS readout, but the open question is whether the early-layer information in C11' survives the layer-1 injection at all. I'm considering a fourth option where the outputs of several layers are combined with pass-through connections.

I'm weighing whether layer 11 could pool from token states that layer 10 had already modified to carry some other signal, but that would still only be deep information. Shallow information would have to travel six or more layers through either the token streams or the CLS state, so the cleaner test is to check whether pooling at an intermediate layer like 6, combined with a shallow one, does better than the late layers.

I'm considering using the layer-6 pooled features as the deep signal, which would free layers 7 through 11 to be repurposed, though the shallow information would still need to be carried forward over a shorter span. I'm also weighing a richer nonlinear readout on the CLS state from layers 7 to 11, though random features didn't help much before.

I'm weighing whether the carried shallow signal could survive as a small perturbation in the residual stream, since BERT-like networks tend to tolerate modest changes. The worry is that any carried supervised score would leak into the other tokens and get distorted, so I'd want to check whether an unsupervised carrier does any better than the supervised direction.

I'm weighing a carry design where the shallow signal is passed forward as a fixed projection of the mean-pooled embeddings, with the final layer reusing the joint ridge weights so no refit is needed. In principle that's exact without perturbation, but the perturbation from the carried channel would make the deep features drift from their fitted values. I can only check it on training data, which doesn't give an honest estimate, and the complexity looks high enough that I'm reluctant to pursue it.

I'm planning an honest check for the carry design: fit the readout weights on half the practice data, evaluate on the other half, then refit on everything. I'm also weighing how sensitive the final layer's accuracy is to perturbation in the carried channel, though I can't know that without trying. Another option I'm considering is isolating the CLS token, which avoids perturbing other tokens if they don't attend to it, so I'd first measure how much the other tokens attend to CLS at each layer.

I'm checking how much the other tokens attend to CLS at each layer, since if that attention is near zero from some layer onward, the CLS state could be changed without disturbing them. If so, its own pooled means from those later layers could accumulate into the final readout.

I'm checking whether summing the pooled features across layers, or running a sum-based classifier on them, holds up as an alternative to concatenation. I'm also noting that the CLS token would need a uniform-attention query to pool consistently, which depends on it being identifiable.

I'm going to check how much the other tokens attend to CLS, so I'll run one pass over the 64 texts and look at every layer.



Token-level probes lose word identity in layer 11 (R² ≈ 0.38 vs 0.55 at layer 6), so the shallow signal has to be carried explicitly. I'll check whether other tokens attend to [CLS] enough that using it as the carrier would leave deep token features undisturbed.



## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "p0014",
  "p0029",
  "p0061",
  "p0073",
  "p0081",
  "p0082",
  "p0084",
  "p0110",
  "p0124",
  "p0142",
  "p0148",
  "p0161",
  "p0184",
  "p0200",
  "p0209",
  "p0215",
  "p0239",
  "p0247",
  "p0262",
  "p0276",
  "p0348",
  "p0352",
  "p0372",
  "p0378",
  "p0390",
  "p0427",
  "p0431",
  "p0438",
  "p0450",
  "p0479",
  "p0482",
  "p0499",
  "p0501",
  "p0516",
  "p0517",
  "p0530",
  "p0536",
  "p0541",
  "p0549",
  "p0568",
  "p0574",
  "p0594",
  "p0600",
  "p0609",
  "p0631",
  "p0657",
  "p0674",
  "p0685",
  "p0694",
  "p0700",
  "p0706",
  "p0717",
  "p0718",
  "p0721",
  "p0727",
  "p0762",
  "p0763",
  "p0767",
  "p0783",
  "p0789",
  "p0797",
  "p0824",
  "p0833",
  "p0838"
 ],
 "outputs": [
  "attention"
 ]
}
```


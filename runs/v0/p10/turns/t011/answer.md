## reasoning summary

I'm settling on about 300+ dimensions for the deep mean stream, since PCA at 256–384 dims keeps around 96% of the variance. That leaves 6 heads for it and 6 for other information. Next I want to check how many PCA dimensions of the layer-0 mean are needed to go along with the layer-9 part.

I'm working out whether the 8 logits can be computed directly inside the network, since a uniform-attention head with V set to the L0 ridge weights would output just those logits. That would need only 8 dimensions, though it depends on the mean-hidden definition matching, so PCA directions with more dimensions may be the sturdier choice. The harder issue is still how layer-0 information gets carried forward, so I'm going to think through a CLS-carrier layout.

I'm now working through a layout where CLS has its query zeroed at layer 0, so its attention output carries the mean of layer-0 hidden states. Other tokens would only see small query leakage at that layer. The real problem is that from layer 1 on, CLS's changed state alters the keys and values other tokens read from it, so I'm weighing whether the CLS representation can be kept looking normal in the early layers.

I'm wondering whether the carrier could be a token nobody attends to, like the final [SEP], but SEP is a heavily attended sink in the middle layers, so that's no good. Maybe the pooling layer could split its heads, with some computing mean(h_a) and others carrying layer-0 token information forward. That only works if the per-token layer-0 information survives in the residual stream, and a linear mean readout doesn't capture the non-linear part that the (0,9) gain depends on.

I'm weighing whether the nonlinear part could come from content-dependent softmax attention in some heads, though it's unclear what in layer 0 is missing from layer 9. Mean(h1) as the shallow stream would hit the same issue, and the carrier question stays open.

I'm considering whether the CLS token could be made invisible to the other tokens from layer 1 onward, so that it can carry the layer-0 mean without disturbing their attention. A constant key bias would cancel out in the softmax, so suppressing CLS would need a key direction that scores low against the typical queries, and I'm working out whether such a direction exists.

I'm now checking whether a constant negative-score direction could hide CLS from other tokens in the middle layers, which would need a CLS-detector at each layer and would leak somewhat. Even with suppression, CLS still evolves through its own attention, so the layer-0 mean gets transformed before it reaches the pooling layer.

I'm weighing whether a shallow pooling could serve as the whole deep stream and let the remaining layers add nonlinear features. But once every token is pooled into one vector there's no per-token context left for those layers to work on, so that doesn't hold up. The carrier approach still looks like a multi-pass research project with a small expected gain, so I'll start with a minimal version and run experiments from there.

I'm weighing a variant where the CLS token alone does a uniform pooling at layer 0 to pick up the bag-of-words signal, with the remaining layers running normally. I'd then check whether CLS's final state picks up that lexical information without much damage to the other tokens. The last layer would still need a separate plan.

I'm now setting the last pooling layer to take a uniform average over the layer-11 tokens into S1 and to read the CLS state into S2 through a CLS-detector key, with the output scaled to dominate the residual and the FFN zeroed out. The final CLS is then a layer-norm of those two projections. For evaluation, I'd run one pass on the modified model to collect the layer-11 CLS and mean states over 1024 items, then fit a cross-validated ridge on their concatenation.

I'm weighing whether to pool at layer 9 instead of 11, since mean(h9) scores a bit higher than mean(h11). In that version the CLS state already carries the layer-0 bag-of-words signal by layer 9, so the last two layers could pass through unchanged.

I'm settling on the final pooling layer's design: uniform heads for the mean, and CLS-detector heads that pull out the CLS state. The idea is to set the query to a constant and steer the key toward CLS so those heads attend only to it.

I'm weighing whether the CLS token can carry the layer-0 mean forward without disturbing other tokens, and the cleanest route looks like skipping attention entirely for the CLS self-route: if the residual itself carries CLS, the only issue is how strongly the pooled mean term is scaled against it, so I'm considering a moderate scale factor c that keeps both contributions in play.

I'm checking whether the CLS and mean signals can be combined linearly: the residual keeps CLS's components outside the pooled subspace, while the pooled subspace mixes the mean with CLS's own contribution.

I'm settling on a simpler design: the pooling heads' value and output projections can carry the mean directly, so the CLS-routing heads aren't needed. The CLS state's leftover components in the complementary subspace should still survive the normalization.

I'm checking the scaling: the CLS residual's component outside the pooled subspace stays around magnitude 20, while the mean term at scale c is about 10c. Even at c=1000, float32 precision holds, but the pooler's tanh over a unit-variance LayerNorm output would shrink those S2 components to roughly 0.2/c, so the classifier would need very large weights on them.

I'm checking whether the scale c can stay moderate, since the CLS residual and the pooled mean term would then both carry signal in S1 and S2. Mixing CLS's own components into S1 is the drawback, and I can't control how x_cls projects onto that subspace. Using LN1's per-dimension gain to separate coordinate-aligned halves of the residual might give the clean split I want instead.

I'm now looking at whether heads 6-11 could route only from CLS, which would mean finding a reliable CLS detector at that layer, and I'm considering whether a positional signal could serve that role instead.

I'm going to start with an experiment before committing to the routing design: one read pass on the layer-0-hacked model, collecting CLS and mean states at every layer over 1024 items. That will show whether the CLS state at the pooling layer carries enough bag-of-words signal, with a cross-validated score of at least 0.97 on the concatenation as the bar. If it falls short, I'll go back to the simpler version.

I'm planning to check the layer-0 hack first by reading CLS and mean states at every layer, and to see how much of the bag-of-words signal survives. If that falls short, I'll fall back to the simpler version. For the hack, I still need a direction that separates CLS from the other tokens, and I'd have to estimate that from frozen embeddings since I don't have token-level states.

I'm considering computing the layer-0 embedding statistics directly from the weights rather than running the model, but that edges toward reimplementing the forward pass, so I'm hesitant. A safer route is to read token-level hidden states at layer 0 for a small set of texts, which would give a margin check without that gray area.

I'm weighing whether I can get the token-level layer-0 states without a reimplementation of the embedding layer, since a read pass would be the safer route. Using the sentence means as a proxy won't give the token-level second moment I need, so the subspace choice is still open.

I'm weighing whether a key-side change could steer attention instead, since the query shift from the CLS embedding leaks into every other token's query and that cost looks large.

I'm considering a different route: instead of changing the query, I could zero out the CLS token's scores by projecting the keys orthogonal to its query direction, which would make its attention uniform. That shifts every other token's scores too, though, so both approaches are rank-one edits with side effects, and I need to see which one costs less.

I'm thinking about computing the token-level layer-0 covariance from the embedding weights, but that would amount to reimplementing the embedding forward pass, so I'll avoid it. Instead I'll read real token-level hidden states at layer 0 for about 64 texts, which gives roughly 2000 vectors, enough for a shrunk 768-dimensional covariance. That costs one extra pass, but it keeps the approach clean.

I'm planning one read pass on the base model over 64 texts, collecting token-level hidden states at layers 0, 8, 9, and 11, plus attention maps across all layers, so I can check how much each token leans on CLS before committing to the layer-0 hack.

I'm weighing whether a read pass over 64 practice texts is worth it, since it would give token-level layer-0 states for estimating the covariance without reimplementing the embedding layer. I'm also collecting attention maps across all layers, so I can see how much each token depends on CLS before committing to the hack.

I'm now weighing a simpler design: pool the CLS state at layer 0 so it carries bag-of-words information, then at a later layer route CLS's own state and the uniform mean into separate subspaces. For heads 6-11 I'd still need a reliable detector that picks out the CLS token, which is the part I haven't settled.

I'm weighing a routing alternative that skips the residual split: at layer a, the CLS state could be routed by a detector direction, though it's unclear whether one fit on the base model would hold for the hacked model. A positional signal is another option, but deep layers may not keep position information reliably, so that's a risk.

I'm going to check how much BoW information survives the split by simulating it on the base model's cls and mean states at layer 9. The mean state would get projected into the top-384 PCA subspace and scaled by c before the layernorm, then compared against plain concatenation with cross-validated ridge. Base-model CLS and mean states alone won't show much BoW signal, so I'll need a different proxy for the layer-0 case.

I'm weighing whether to skip a separate pass for the layer-0 covariance, since a token-level read on the base model would give the vectors I need, and the same pass could also show how much each token leans on CLS across layers. That's eight passes left, so I can afford one read on 64 texts before committing to the hack.

I'm deciding how to pick the 64 texts for the token-level read. Rather than the first 1024 and their labels, I'll take every 16th id so the sample is spread out, and I'm still unsure whether the layer and attention settings apply to both outputs.

I'm settling on requesting every layer for both the token-level hidden states and the attention maps, rather than a trimmed subset. The attention arrays only cover encoder layers 1 through 12, so picking specific indices would leave layer 0 out of that output anyway. The total comes to roughly 315MB, which should be fine on disk, so I'll keep the default and not complicate the request.



The layer-0 [CLS] vector is the same for every sentence, and mean features need 300+ dimensions. I'm reading token-level states and attention on 64 texts next to build a [CLS] detector direction.



## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "p0000",
  "p0016",
  "p0032",
  "p0048",
  "p0064",
  "p0080",
  "p0096",
  "p0112",
  "p0128",
  "p0144",
  "p0160",
  "p0176",
  "p0192",
  "p0208",
  "p0224",
  "p0240",
  "p0256",
  "p0272",
  "p0288",
  "p0304",
  "p0320",
  "p0336",
  "p0352",
  "p0368",
  "p0384",
  "p0400",
  "p0416",
  "p0432",
  "p0448",
  "p0464",
  "p0480",
  "p0496",
  "p0512",
  "p0528",
  "p0544",
  "p0560",
  "p0576",
  "p0592",
  "p0608",
  "p0624",
  "p0640",
  "p0656",
  "p0672",
  "p0688",
  "p0704",
  "p0720",
  "p0736",
  "p0752",
  "p0768",
  "p0784",
  "p0800",
  "p0816",
  "p0832",
  "p0848",
  "p0864",
  "p0880",
  "p0896",
  "p0912",
  "p0928",
  "p0944",
  "p0960",
  "p0976",
  "p0992",
  "p1008"
 ],
 "outputs": [
  "token_hidden",
  "attention"
 ]
}
```


## reasoning summary

I'm checking whether pruning to 5 layers keeps test accuracy close enough: it agrees with the base model on 95% of items overall and 82% of low-margin items, which look more like the test set. Base test accuracy is 77.5% and the target is 76.5%, so I'm estimating how many disagreements would have to be errors to hit that bar.

Pruning to 5 layers would be hard to make work: with about 95% agreement with the base model, the disagreements could cost several points of test accuracy, and the target allows only a 1-point drop. I'd need the pruned model's disagreements to be almost entirely cases where the base was wrong, which seems unlikely. Plain truncation looks too lossy, so I'm weighing alternatives: narrowing the FFN across all 6 layers, or dropping a single layer chosen by testing each candidate.

I'm weighing whether dropping a different layer, rather than the last one, would hold up better, and I'd have to test each candidate. Drop 0 is probably catastrophic, but I'd want to confirm that, and the earlier truncation result doesn't transfer directly once the head is refit. Compensating for a dropped layer by folding a linear map into the neighboring layer's normalization parameters is one idea, though it only partly fixes the mismatch.

I'm now looking at a cheaper compensation trick: fitting a per-dimension diagonal affine map from one layer's output to the next and folding it into the previous layer's LayerNorm gamma and beta. That's fully foldable, though it can only correct scale and offset per dimension, not the full linear mismatch.

I'm considering whether a diagonal affine correction per dimension, folded into the previous layer's LayerNorm, can recover some of the mismatch from dropping a layer, since post-LN residual streams often correlate strongly across adjacent layers. A full linear map would only fit into the attention inputs of the next layer, and the FFN input still carries the residual, so the diagonal version is the one I can fully fold. The mean shift looks like the largest discrepancy, which an affine term would absorb. Before committing, I'd want to measure how the base logits move under each edit on the low-margin items.

I'm checking the test size, since 77.5% of about 400 examples is roughly 310 and the target means I can afford to lose only around four. Macro-F1 is also dropping from 0.71 to 0.69, which may be hurting because the governance class is small. Any pruned model will disagree with the base somewhere on test, and that noise could easily exceed the margin.

I'm weighing the compute-per-agreement tradeoff now: dropping one layer saves about 7.09M parameters, which is roughly the same as narrowing the FFN across six layers to about m=2304. Within six layers, narrowing to m=1536 would save about 14.2M, so the question is which option keeps agreement with the base model high enough on low-margin items.

I'm planning to collect token-level hidden states for about 32 examples across all layers, picking mid-length items and a range of classes. I'll compute FFN unit importance from those, since that will guide which neurons to prune, and I'll also keep the diagonal-affine compensation idea in mind for any dropped layer.

I'm going to estimate the FFN unit importance from how each unit's activation deviates from its mean and how much it pushes into the output. The next step is to measure that per unit so I can decide which neurons to remove.

I'm working out how to fit the FFN units for a joint least-squares compensation: the kept units' outputs would be refit to match the full layer's output over tokens. That needs the activations, which I'd have to compute from the proxy inputs, and I'm questioning whether that is just running part of the forward pass.

I'm checking whether a Gaussian approximation for the expected GELU output is closed-form enough to use for bias compensation, since running the actual activations would count as forward computation.

I'm checking whether the Gaussian closed form for the expected GELU output holds up, since the pre-activations may not be Gaussian for units that fire on specific tokens.

I'm planning a few pruning runs: six layers at m=1536 with importance-based unit selection and bias compensation, checking agreement with the original head both with and without a refit, and then trying smaller widths.

I'm planning a second pass to gather token-level stats on about 32 examples, stratified by class and limited to 80–200 tokens, so I can estimate which FFN units matter most and whether a dropped layer can be compensated.



Truncating to 5 layers with a refit head still disagrees with the base model on 5% of practice items, which matters since the test margin is only one point. Next I'm gathering token-level hidden-state statistics across layers to drive unit importance and layer-drop compensation, using 32 class-stratified samples.



## tool_use: run_python

```json
{
 "code": "P=practice(); y=np.load('y.npy')\ntoks=tokenize([p['text'] for p in P]); L=np.array([len(t['ids']) for t in toks])\nrng=np.random.default_rng(1)\nsel=[]\nfor c,nc in zip(range(4),[6,8,10,8]):\n    cand=np.where((y==c)&(L>=90)&(L<=200))[0]\n    sel+=list(rng.choice(cand,nc,replace=False))\nsel=sorted(sel); print([P[i]['id'] for i in sel], L[sel].sum(), L[sel].max())\njson.dump([P[i]['id'] for i in sel],open('tok_ids_A.json','w'))"
}
```


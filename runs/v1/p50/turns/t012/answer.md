## reasoning summary

I've got late fusion of the 8-dim logits from m0 and m11 at 97.3%, close to the joint 97.5%, so I only need to carry those 8 ridge logits from layer 0 through to the end. I'm now working out how to pass 8 values per token from layer 0 to the final aggregation, since the logits are linear in the tokens and I need a per-token value for them.

I'm weighing two ways to carry those 8 values: a per-token carry in reserved coordinates across the 11 layers, or aggregating them into CLS early and carrying them there. Either way I'd need coordinates that later layers neither read nor write, which means zeroing the corresponding weight columns and accepting that the original content of those coordinates is lost.

I'm weighing a per-token carry in reserved coordinates against a CLS carry. Either way, I'd have to zero the reads and writes on those coordinates from layer 1 on, which costs a little of the residual stream everywhere. I'm checking whether leaving the reads intact would be less disruptive.

I'm weighing whether to repurpose some of layer 0's FFN units to write the 8 values, since losing 16 of 3072 units would be a small cost. The catch is that the FFN input is the layer-normalized attention mix, so I'd have to account for that before writing anything into the reserved coordinates.

I'm looking at whether the embedding's reserved coordinates could hold the signal directly, since the embeddings are frozen and their reserved dimensions contain whatever they happen to hold.

I'm weighing whether layer 0's FFN could write the 8 values directly, though the input there is the layer-normalized attention mix rather than the raw embedding. Freezing the reserved embedding coordinates would be simpler, but those raw coordinates carry far less discriminative signal than a fitted 8-dim projection, so I'm leaning toward writing the projection itself.

I'm working out how to write the 8 projected values into the reserved coordinates of layer 0's FFN: a ± unit pair gives an exactly linear term, so the output stays linear in the attention mix. I still need to cancel the existing content of those coordinates in the residual, which the next steps cover.

I'm checking whether the 8 projected values can be written into layer 0's FFN as an exactly linear term, using a ± unit pair per value. The open question is how to fit that projection, since the layer-0 post-attention state isn't directly readable.

I'm trying to read the layer-0 post-attention state directly, but the read API only exposes hidden states at layer boundaries, so that's not an option. One workaround I'm weighing is zeroing layer 0's FFN output temporarily so the residual passes through unchanged, which would let me pull out a usable version of that state for fitting the projection.

I'm considering whether to fit the projection on the embedding outputs and write it through layer 0's attention, but attention mixes across tokens, so that doesn't work cleanly. Since carrying from layer 1 or 2 already gives nearly the same accuracy, I'm thinking about shifting the carry to one of those earlier layers instead.

I'm considering whether an attention head that attends only to itself could carry the projection of each token's own state, since that would give a linear write without going through the FFN's post-attention input.

I'm considering whether a head that attends only to itself could carry each token's own projection, which would give a linear write without going through the FFN's post-attention input. The catch is that making a head attend to self depends on its query-key scores, and repeated tokens or similar positions could split attention, and it would cost that head's original role.

I'm considering a different option: a head in layer l with its query zeroed would attend uniformly, so its output at every token is the mean of V applied to the hidden states, which is the sentence-level aggregate I want. I'd fill only 8 rows of its value matrix with the projection A and route that head's output-projection columns into the reserved coordinates R, accepting that the head's original role is lost.

I'm now checking whether zeroing the reserved coordinates' LN gain and bias at layer l−1's output would clear the residual content there, so the carry starts clean. The cost is that those coordinates would be zero as input to layer l, and I'd also need to fit A on states where R is already zeroed.

I'm planning to use a head in layer l that attends uniformly, with its value matrix carrying the projected states into the reserved coordinates, while the other heads still write normally across the full residual.

I'm checking how to zero the reserved rows of the attention output projection so only head H writes there, then clearing the bias and setting the LayerNorm gain and bias for those coordinates so they carry only the mean-aggregated feature.

I'm now zeroing the reserved coordinates' reads in the later layers and their writes in the FFN and LayerNorm outputs, so the carried signal stays isolated there.

I'm weighing whether to zero the reads of the reserved coordinates in the later layers, since any sentence-level signal sitting there would perturb every token's computation. Leaving them readable would feed the model values it never saw in training, so zeroing them is probably the cleaner choice, even though the base model's token-level information in those coordinates is lost either way.

I'm checking how the sentence-level signal in those reserved coordinates scales through the LayerNorms, since its magnitude shifts per token and only averages out at the end. I'm also considering whether the aggregation at the final layer, with uniform attention and an identity value matrix, would reproduce the mean cleanly.

I'm checking whether layer 0 can clear the embedding content in those reserved coordinates, since the embedding LayerNorm is frozen. Zeroing the attention contribution won't remove what the embedding already put there, so I'm weighing whether to use layer 0's LayerNorm gain to suppress them and write the signal through the FFN instead.

I'm checking whether layer 0 can carry the signal too: the residual there still holds embedding content in those coordinates, but that per-token part just averages into the final features, so it doesn't need cancelling.

I'm checking whether layer 0's residual, which still holds per-token embedding content in the reserved coordinates, can be used as is. Averaging over tokens would mix that per-token part with the attention-written projection, so I'm considering setting the head's value matrix to the projection minus the reserved-coordinate selector, so the total at the end comes out as the desired projection.

I'm weighing whether to start the carry at layer 1 with the reserved coordinates' gain and bias zeroed at the previous layer's output, since layer 0's frozen embedding content would otherwise leak into the mean. Folding that leftover per-token part into the value matrix could cancel it, but the FFN reads at layer 0 would then pick up the signal, which costs some information.

Considering whether to keep the 8 reserved coordinates at layer l rather than pushing the bag-of-words signal through the full stack. Carrying the 8-dim ridge logits from layer l with the final m11 features could land near 97.3%, and it's worth testing whether layer 2 does better than layer 1.

I'm now weighing the overfitting risk of fitting the projection D on all 1496 practice examples, since that could make the late-fusion weights look better than they will be on held-out data.

I'm weighing how to fit the final classifier without overweighting the early-layer features, since their in-sample ridge logits look overconfident on the training data. I'm planning to choose the fusion weight by cross-fitting, as the earlier CV run did, and then fit the late-layer ridge on the features that exclude the reserved coordinates.

I'm now writing the final classifier as two parts: a ridge fit on the non-reserved coordinates of the final CLS features, plus a decoding of the reserved coordinates into logits with weight α. The reserved values are scaled per sentence by the LayerNorm factors along the path, so I still need to account for that.

I'm checking how the final-layer aggregation would read the scaled reserved coordinates, and whether the per-sentence scale factors stay roughly constant enough to keep the fusion weight stable.

I'm comparing late-fusion cross-validation for carrying the 8-dim ridge logits from several layers (m0, m1, m2, m3, m4) alongside the final-layer features, and I'm also weighing whether a 16-dim carry over two layers is worth it, though I'll keep it simple for now.

I'm considering whether layers beyond the first two could also carry reserved-coordinate logits, which would mean more reserved dimensions and more damage to the model. I want to pick the reserved coordinates by looking at their per-coordinate statistics, avoiding the known high-magnitude outlier dimensions and favoring low-variance ones.

I'm checking whether the reads of those reserved coordinates in the later layers can simply be cut. Zeroing their columns in the query, key, value and intermediate weights of each later layer would stop the signal from being read, while layer l itself can keep its attention as is, since its input there is already zeroed. Only the layer-l FFN would need its intermediate columns for those coordinates zeroed, because it reads the signal directly.

I'm estimating how the LayerNorm scaling affects the reserved coordinates, using rough guesses for the mean and standard deviation of the hidden vectors, with the gain set to 1 for now.

I'm now looking at whether the mean that layer-l's LayerNorm subtracts adds per-token noise to the reserved coordinates, which could matter once it's accumulated across all the layers.

I'm noting that a shared shift across all 8 reserved logits leaves the argmax unchanged, and a uniform per-token scale does the same, so the carried vector is only rescaled and offset. What I still need to settle is whether giving the 8 reserved gains equal values keeps that property intact.

I'm setting the head's value rows for the reserved coordinates to the projection D, zeroing the reserved columns of the input, and making the output projection map those rows to an identity over the eight reserved dimensions. Everything else in that output projection stays zero for those rows, and I still need to handle the non-reserved rows.

I'm weighing whether to sacrifice one of layer l's heads to build the uniform aggregation, since its original attention pattern can't be kept alongside it. My leaning is to pick a head whose attention is already broad, so that replacing it costs the least.

I'm considering whether to keep the head's original value and output paths for the non-reserved coordinates and free up eight of its value dimensions for the projection D, by truncating the head's output via SVD to rank 56. That would keep most of its behavior intact, provided I pick a head whose attention is already close to uniform.

I'm weighing whether to drop one head from layer 11 instead of repurposing the whole layer, since removing a single head out of 144 should barely change the model.

I'm now thinking about the final layer: since CLS has to attend uniformly to every token there, layer 11 can be repurposed wholesale to compute the mean of the token states, with its FFN serving as an extra nonlinear step on top. I'm also considering carrying reserved logits from two layers, layer 2 and layer 6, which would double the reserved coordinates to 16.

I'm going to test several late-fusion options numerically, pairing the final layer with early or middle layers in various combinations, and also compare against a joint ridge on the full layer-l features as an upper bound. I'll run the fusion cross-validation for each option.



Late fusion of the two 8-class ridge heads reaches 97.3%, close to the joint 97.5%, so I'm designing a way to carry just those 8 logits through to the end and checking which early layer to take them from.



## tool_use: run_python

```json
{
 "code": "lab = np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nM=np.load(WORK/'base_mean.npy')\ndef fit_ridge(Xt,Yt,lam):\n    mu=Xt.mean(0); sd=Xt.std(0)+1e-6; Z=(Xt-mu)/sd\n    W=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Yt-Yt.mean(0)))\n    return lambda X: ((X-mu)/sd)@W\noof={}\nfor l in range(13):\n    for lam in [100,300]:\n        o=np.zeros((N,8))\n        for f in range(5):\n            tr=folds!=f; te=~tr\n            o[te]=fit_ridge(M[l][tr],Y[tr],lam)(M[l][te])\n        oof[(l,lam)]=o\nnp.save(WORK/'oof_ridge.npy', np.stack([oof[(l,300)] for l in range(13)]))\nacc=lambda S:(S.argmax(1)==lab).mean()\nfor l in range(13): print(l, round(acc(oof[(l,100)]),4), round(acc(oof[(l,300)]),4))\nprint('fusion pairs with 11 (lam 300 early, 100 late):')\nfor l in range(0,11):\n    print(l, [round(acc(w*oof[(l,300)]+oof[(11,100)]),4) for w in [0.5,1,1.5]])\nfor combo in [(1,6),(2,6),(1,5),(2,5),(1,4),(2,8),(1,8),(0,6)]:\n    print(combo, round(acc(oof[(combo[0],300)]+oof[(combo[1],300)]+oof[(11,100)]),4))\n"
}
```


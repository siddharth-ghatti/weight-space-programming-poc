## reasoning summary

The joint-ridge h0 directions reach about 97% stacked, and 8 carried dimensions capture most of that gain. Now I'm working out how to choose D for the carry: looking for residual-stream dimensions with low variance across tokens and low use by the weights, using per-dimension stats of the hidden states.

Next I'm weighing which residual dimensions to set aside for the carry, favoring ones with low variance across tokens and little reliance from the weights. For the early layers, the plan is to zero out one dimension's contribution in the attention output and layer norm, then repurpose a set of low-importance FFN neurons to read the clean dimensions and fire on a large positive bias.

I'm working out the rest of the layer-0 edit: after the FFN output, the LN2 gain for the carried dimensions and the offset that keeps them centered. The main worry is that the mean subtraction in the LayerNorm also shifts the carried values, so I want the signal scale small enough that it doesn't disturb the normalization statistics.

I'm worried that the LayerNorm mean subtraction leaves a shared offset across all eight carried dimensions, which would act as a nuisance direction. I'm considering whether the final ridge can handle it by reading the carried signals through contrasts orthogonal to the all-ones vector over those dimensions.

I'm checking the ridge readout now: since the eight class targets are centered, the readout weights sum to zero across the carried dimensions, which leaves effectively seven directions that are orthogonal to the all-ones offset from LayerNorm. So the shared nuisance along that direction can simply be projected out.

I'm checking how the LayerNorm offset behaves on the carried dimensions: with unit gain and zero bias, each one picks up a shared shift along the all-ones direction, and the signal itself only gets scaled by the per-token inverse standard deviation. Since that inverse standard deviation depends on the pre-LN residual scale at each layer, I need to keep track of how large those inputs are.

I'm working out how the D dimensions' scale would decay through the LayerNorms: with unit gain, each LN divides by the token's standard deviation, which would shrink the signal geometrically across roughly 22 norms. The fix I'm considering is setting each D-dimension's LN gain to about the typical standard deviation at that layer so the magnitude is preserved on average, leaving only a per-token multiplicative variation.

I'm weighing whether the LayerNorm scaling on the carried dimensions can be kept stable, since each LN divides by a per-token standard deviation that I can't measure without a forward pass. The per-token variation in those ratios would make the carried signal a weighted average rather than a plain mean, which could hurt the ridge readout.

I'm still weighing how the LayerNorm scaling affects the carried signal: each norm divides by a per-token standard deviation, so the signal could end up as a token-weighted sum rather than a plain mean. I'm considering whether that reweighting, if it's correlated across tokens within a sentence, would still leave the ridge readout usable.

I'm weighing whether the layer-norm scaling on the carried dimensions can stay stable across the stack: each norm divides by a per-token standard deviation, so the carried signal would become a token-weighted sum. Rather than solving that analytically, I'll check it by experiment, since the final stage may tolerate a sentence-level scale factor well enough.

I'm weighing a way to keep the carried signal from being distorted through the remaining layers: the layer norms would rescale it per token, and I'd rather avoid that. Carrying the dimensions through the rest of the network seems hard to make clean, so I'm leaning toward accepting the per-token weighting and checking how much it actually hurts the readout. I also still need to protect those dimensions from being read by the later layers, starting with the original embedding in layer 0's attention.

I'm setting the carried dimensions' LayerNorm gain and bias in layer 0's attention output to zero, and noting that the FFN will read slightly zeroed inputs there, which costs a little. After that the FFN output for those dimensions gets cancelled by a bias term, with the second LayerNorm gain set to match the carried magnitude.

I'm now setting the carried dimensions to zero in the attention and FFN reads and writes across layers 1 through 10, and resetting their LayerNorm bias to zero with the gain held at the chosen level. I'm also weighing which dimensions to sacrifice, leaning toward ones with small LayerNorm gains or low measured importance so the damage to the original model stays small.

I'm now looking at the final layer: uniform attention with an identity-like value path, so the carried features become a mean over tokens. The worry is that the LayerNorm mean subtraction adds per-token offsets along the all-ones direction on those dimensions, and I'm checking whether they matter because BERT hidden states contain some large-magnitude dimensions.

I'm settling on the scale of the carried signal: I want its per-dimension spread to be close to the typical spread of the hidden states, so it doesn't distort the LayerNorm statistics, while still standing out enough in the final normalized mean to be read.

I'm wondering whether the signal magnitude matters: the carried dimensions add to each token's normalization only slightly, so that effect looks small. The open question is how to choose each dimension's LayerNorm gain, since I can't measure the per-layer standard deviation without a forward pass. If I leave the gain at one, the signal decays across the layers, and I'm trying to decide whether that would make the final readout unusable.

I'm wondering whether the D dimensions can be read reliably at all, since their values would be around 1e-4 relative to typical activations, leaving only about three significant digits in float32. The ridge standardizes columns so magnitude alone shouldn't matter, but that precision is a real risk. I think I need to estimate the per-layer standard deviation before deciding on the LayerNorm gains.

I'm planning a third pass that reads the edited model across all 1024 ids and records the D-dimension magnitudes at each layer, so I can see how fast the signal decays. If the decay is severe, I'd adjust the per-layer LayerNorm gains from the observed rates and run another pass. I'm also wondering whether I can estimate the typical standard deviation at each norm directly, since BERT hidden-state norms might give a starting point.

I'm estimating the per-layer LayerNorm scale for the carried dimensions, starting from a guess that BERT's per-dimension spread is around 0.8 to 1.5, which would shrink the signal by roughly 0.02 across 22 norms. I'll start with a gain of one and correct it from the observed decay in the next pass. I'm also considering setting each carried dimension's gain from the average gain of the non-carried dimensions, since the residual stream roughly preserves magnitude.

I'm trying to estimate the per-layer standard deviation, which would set the LayerNorm gains for the carried dimensions, since I can't read σ directly from the hidden states. One idea is to match the carried signal's scale to the residual stream: if the stream keeps roughly constant magnitude, the gain should track the normalization's σ. I'd rather settle that empirically than guess it.

I'm settling on empirical measurement, since I can't estimate the per-layer scale from the weights alone. I'll look at whether the original gains for the carried dimensions would track the content, but the decay itself has to be observed from an actual forward pass.

Checking whether the attention mask excludes padding in the final layer, and whether the [CLS] and [SEP] states, which look like they have large norms, would distort the uniform average. I'm considering dropping them from the pooling, though a constant-ish offset scaled by 1/T may be harmless.

I'm going to pull together per-dimension statistics before picking the carried dimensions: the mean hidden values per layer and their variance across sentences, the LayerNorm gains for each dimension, and the norms of the attention and FFN weights that read and write each dimension.

I'm ranking the dimensions by an importance score, roughly the read-weight norms times each layer's typical activation size. Since token-level magnitudes aren't available, I'm using the sentence means and the CLS states as a proxy. I'm also looking at the 8 FFN neurons in layer 0 with the smallest output norms as candidates to repurpose.

I'm going to pull the statistics together now, starting with the eight layer-0 FFN neurons that have the smallest output norms and so are the cheapest to repurpose.



Layer-0 directions from the joint ridge reach about 97.0% stacked on the mean, so 8 dims carry most of the gain. Next I'm picking the 8 residual dimensions to sacrifice, favoring those with weak BERT weight reads and writes and small activations, plus the 8 weakest layer-0 FFN neurons.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\npre='bert.encoder.layer.'\nreadn = np.zeros(768); writen=np.zeros(768); gam=np.zeros((12,2,768))\nfor l in range(12):\n    for k in ['query','key','value']:\n        readn += np.linalg.norm(base(f'{pre}{l}.attention.self.{k}.weight'),axis=0)\n    readn += np.linalg.norm(base(f'{pre}{l}.intermediate.dense.weight'),axis=0)\n    writen += np.linalg.norm(base(f'{pre}{l}.attention.output.dense.weight'),axis=1)\n    writen += np.linalg.norm(base(f'{pre}{l}.output.dense.weight'),axis=1)\n    gam[l,0]=base(f'{pre}{l}.attention.output.LayerNorm.weight'); gam[l,1]=base(f'{pre}{l}.output.LayerNorm.weight')\nmag = np.zeros(768)\nfor l in range(1,12):\n    c = load('cls',l); m = load('mean',l)\n    mag += np.sqrt((m**2).mean(0)) + np.sqrt((c**2).mean(0))\ngmean = np.abs(gam).mean((0,1))\nscore = readn/readn.mean() * mag/mag.mean()\no = np.argsort(score)\nprint('lowest score dims', o[:20])\nfor d in o[:12]: print(d, round(score[d],3), round(readn[d],2), round(writen[d],2), round(mag[d],2), round(gmean[d],3))\nprint('typical: readn', readn.mean(), 'writen', writen.mean(), 'mag', mag.mean(), 'gamma', gmean.mean())\nprint('outlier mags', np.argsort(-mag)[:5], np.sort(mag)[-5:])\n# layer0 FFN neurons\nW2 = base(pre+'0.output.dense.weight'); b1 = base(pre+'0.intermediate.dense.bias')\non = np.linalg.norm(W2,axis=0); print('ffn0 out norms min/mean', np.sort(on)[:10], on.mean())\n"
}
```


## reasoning summary

Carrying from layer 1 or 2 seems to get about 97.2 to 97.3 percent, and a second carry doesn't add anything. I'm leaning toward carrying from l=1: I'd zero out R at layer 0's output LayerNorm, then let layer 1 do the aggregation, with layers 1 through 10 ignoring R and layer 11 doing the final aggregation. I'm now checking how the count works when l=1.

I'm working out which hidden state index corresponds to which encoder layer: hidden_states[1] is the output of layer 0 and the input to layer 1, so carrying m1 means wiping R in layer 0's output LayerNorm and using layer 1's head. I'm also noting that the m11 features will shift from the base model, so I'll need to check the degradation, and I'm leaning toward picking the 8 least important coordinates by their variance and LayerNorm scale.

I'm weighing which coordinates to zero out: the ones whose downstream reads contribute least, roughly |h[R]| times the column norms of the read weights. Outlier dimensions like 308 and 381 are out, and I'm considering a score built from per-coordinate spread across sentences in the mean and CLS features, weighted by those reading-weight norms, to pick the 8 smallest.

I'm picking the 8 coordinates to zero out by a proxy: their typical magnitude times the norms of the weights that read them, since that's roughly how much each one feeds downstream. The per-token stats I have are only sentence means and CLS samples, so a fresh pass over 64 texts for true per-token RMS might be worth it, but I'll try the proxy first, build the edit, and measure.

Now I'm settling how to handle the attention heads in layer 1: rather than working through attention patterns, I'm leaning toward simply sacrificing one head outright.

Now I'm setting the final layer up so attention is uniform, with the value and output projections reduced to identity-like maps that pass the mean of the last hidden states through.

Setting the final layer's attention to uniform and its value and output projections to pass-through leaves the CLS output as a scaled mean of the token states plus the CLS state. I'm weighing whether a scale of 1 keeps the CLS signal useful, since the linear probe then sees only the combined sum.

I'm weighing whether to pick the 8 coordinates to zero out via a combined CV-based ridge on the sum of the scaled mean and CLS features, since the LayerNorm step would be my own reimplementation of a forward-pass piece. Keeping the final fit on actual read activations from the edited model feels safer, so the selection of c could just evaluate the unnormalized sum.

I'm settling the pooler layer: the first eight rows would carry the scaled final classifier weights on the CLS output, with the bias scaled to match and the remaining rows zeroed. Since tanh is monotone, the argmax ordering should survive, though I still need to check that a tie doesn't break it.

Next I'm wiring up the edits: zeroing the scale and shift for the carried coordinates in layer 0's output LayerNorm, then blanking out one attention head in layer 1 so it only passes a fixed projection of the carried values.

Next I'm zeroing the carried coordinates in the attention and feed-forward weights of layers 2 through 10, and resetting their LayerNorm shifts to zero while keeping the scales at the layer-1 values.

Now I'm settling the scale of the value logits in layer 1, starting with γ_R at 1 and checking how the LayerNorm divisions will shift them.

I'm planning to tune the amplitude of the carried coordinate at each layer's LayerNorm, starting from a guess that the typical scale is around 1. Rather than rely on that guess, I'll read the per-layer activations on those coordinates after one pass and adjust the gains to keep the magnitude near unit size. I'm also going to be checking how the intercept and standardization of the layer-1 probe feed into the logits.

Weighing layer 1 against layer 2 for the carry point, the two look about equal, so I'm going with l=2: wipe the carried coordinates at layer 1's output LayerNorm and aggregate in layer 2. I'm still checking the ridge regularization, since the standalone layer-2 result at lambda=300 comes in below lambda=100.

I'm going to score each coordinate by how strongly the downstream layers read it, combining its typical activation size with the norms of the weights that read it, and then pick the eight smallest.



Carrying from layer 2 gives about 97.2–97.3% CV, with no gain from a second carry. Next I'll pick the 8 least important reserved coordinates and the layer-2 head to sacrifice.



## tool_use: run_python

```json
{
 "code": "M=np.load(WORK/'base_mean.npy'); C=np.load(WORK/'base_cls.npy')\nscore=np.zeros(768)\nfor l in range(2,12):\n    rms = np.sqrt(0.5*(C[l]**2).mean(0)+0.5*(M[l]**2).mean(0))  # proxy for token rms\n    pre=f'bert.encoder.layer.{l}.'\n    cn = sum(np.linalg.norm(base(pre+n),axis=0) for n in ['attention.self.query.weight','attention.self.key.weight','attention.self.value.weight'])/3 + np.linalg.norm(base(pre+'intermediate.dense.weight'),axis=0)/2\n    score += rms*cn\norder=np.argsort(score)\nprint('lowest', order[:20], score[order[:20]].round(2))\nprint('highest', order[-10:], score[order[-10:]].round(2))\nprint('median', np.median(score).round(2))\nrms11=np.sqrt((C[11]**2).mean(0)); print('top rms dims L11', np.argsort(rms11)[-5:], np.sort(rms11)[-5:].round(2))\n# head contribution in layer 2: ||W_o[:,H] W_v[H,:]||_F weighted by input scale\nWo=base('bert.encoder.layer.2.attention.output.dense.weight'); Wv=base('bert.encoder.layer.2.attention.self.value.weight')\nrms2=np.sqrt(0.5*(C[2]**2).mean(0)+0.5*(M[2]**2).mean(0))\nhc=[np.linalg.norm(Wo[:,64*h:64*h+64]@(Wv[64*h:64*h+64,:]*rms2[None,:])) for h in range(12)]\nprint('layer2 head contrib', np.round(hc,2))\n"
}
```


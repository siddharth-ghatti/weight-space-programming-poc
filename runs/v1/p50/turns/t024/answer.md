## reasoning summary

The remaining errors look genuinely ambiguous, like the cases about notice duties versus claims procedure. At 97.8% the result is solid, and better models might add a little more, though it's hard to separate that from noise. I'm leaning toward the joint approach with carry heads at layers 2, 6, and 10, since the candidate sets all score almost the same and this one sacrifices the fewest heads.

I'm settling on carry heads at layers 2, 6, and 10 over the other candidate sets, since they all land within noise of each other and this one sacrifices the fewest heads. For the implementation, I'm planning to prepare the model by zeroing the query, value, and output columns for the sacrificed heads at layers 6 and 10 while leaving the carry rows ready to receive the signal.

I'm reducing the carry gain so the carried values stay small, which keeps the layer norm statistics nearly unchanged for the other coordinates. The per-sentence scale variation is small enough that I could either accept it or refit the late layers on the actual features after installing the carry heads.

I'm now working through the joint approach's implementation details. The trickier question is whether the carry-layer heads change the layer norm statistics for the other coordinates, which would shift the non-carried states slightly. I'm planning to read the prepared model's mean hidden states at layers 2, 6, and 10, since those are the inputs to the carry heads.

I'm working out how to read the activations for the prepared model: the mean hidden states at layers 2, 6, and 10 excluding the carry coordinates, plus the CLS state at layer 12. That takes two passes. Layers 0 and 1 are untouched, so the base layer-2 states match the edited model, while layers 6 and 10 shift because of the ablations and the sacrificed heads.

I'm setting the sacrificed heads at layers 6 and 10 as well, so all three carry layers are ablated before I read the activations. For the sacrifice I'm choosing the heads at 6 and 10 with the smallest contribution norms, then I'll read the mean states at layers 2, 6, and 10 plus the CLS state at layer 12. After that I'll fit the joint ridge at lambda 300 on standardized features and split the weights into per-layer pieces for the carry heads and the late layers.

I'm writing the pooler and classifier weights so the carried coordinates get decoded with the inverse gain, while the late-layer weights stay on the non-carried coordinates.

I'm weighing whether the sensitivity of the late layers to the scaled carry signal matters, and I'm keeping the carry gain small so the layer norm statistics for the other coordinates barely move. Next I'm checking the verification pass on the practice set after everything is installed.

I'm updating the build script so it supports several carry layers, with per-layer carry weights, a gain parameter, and sacrificed heads at layers 6 and 10. For the prepared read I'm weighing whether the layer-2 head should keep its old weights or also be zeroed, since either choice changes the layer norm statistics a little.

I'm also noticing that the sensitivity worry is covered, so I'm turning to the prepared read itself: the carry layers have to be ablated before the activations are taken, and since the layer-2 head is already sacrificed, I'll zero the heads at layers 6 and 10 too. For the prepared model I'll set the carry weights to zero, so the carried coordinates stay empty at read time. Next I'm computing the contribution norms of the heads at layers 6 and 10 to pick which ones to sacrifice.



I'm settling on a joint design with carry heads at layers 2, 6 and 10, and next I'm picking the least important heads in layers 6 and 10.



## tool_use: run_python

```json
{
 "code": "M=np.load(WORK/'base_mean.npy'); C=np.load(WORK/'base_cls.npy')\nfor l in [2,4,6,8,10]:\n    Wo=base(f'bert.encoder.layer.{l}.attention.output.dense.weight'); Wv=base(f'bert.encoder.layer.{l}.attention.self.value.weight')\n    rms=np.sqrt(0.5*(C[l]**2).mean(0)+0.5*(M[l]**2).mean(0))\n    hc=[np.linalg.norm(Wo[:,64*h:64*h+64]@(Wv[64*h:64*h+64,:]*rms[None,:])) for h in range(12)]\n    print(l, np.round(hc,2), 'min head', int(np.argmin(hc)))\n"
}
```


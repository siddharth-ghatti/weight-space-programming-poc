## reasoning summary

The test error is running roughly 1.3 to 1.7 times the training error, so the model is somewhat data-limited. Doubling the data might trim test error by about 10 to 15 percent, and lambda barely matters, with 0.1 doing a bit better. The L2 test number here (.0715) doesn't match the earlier run (.0589), but that's likely just a different random split.

I'm weighing whether a fixed-size model can do better than it does now. Smarter unit selection, re-fitting the kept units' input weights, or merging dropped units into correlated kept ones each run into the same limits: the nonlinearity blocks closed-form fixes, and merging is just least-squares compensation again. So at this size, the realistic gain is the modest one more data would bring.

I'm now looking at whether layer 5's FFN could be made accurate on just the CLS position with fewer units, since only CLS outputs matter there. The question is what error layer 5 contributes at different unit counts, and whether a smaller set could cover the CLS input distribution well.

I'm weighing whether the test flip rate can be estimated from the margin distribution rather than measured directly. On practice, the base model is wrong on about 7.5% of items, often with a fairly large margin, while margins under 0.5 account for about 5.5%, so I'd guess the test flip rate lands somewhere around 10 to 12%.

I'm estimating the test flip rate by summing, over test items, the chance that noise exceeds each item's margin. Practice flips look like a fair proxy because the compensation was fit on only a small subset of sequences, so most of the practice data is effectively held out.

I'm estimating the test flip count at about 15 for this model size, which puts the chance of coming out worse than the base somewhere around 15 to 20 percent. That's high enough that I'm now considering a larger model, so the margin-based estimate is the number that matters.

I'm checking the chance that the smaller model comes out worse than the base on test: at about 15 expected test flips, the loss-side tail for net accuracy dropping below the threshold comes to roughly 19%. A model with about 8 test flips would have a much tighter spread, so I'm now weighing how much that reduction in risk is worth against the larger model's cost.

Weighing a smaller model against the scoring: under a linear reduction score, the 1536 choice still comes out ahead despite its roughly 19% risk of failing, so I'm now asking whether going even smaller than 1024 could push the expected score higher still.

Weighing the m1280 option, I'm estimating about 30 test flips and roughly a 35% chance of failing, which would give an expected score near 0.13. Under linear scoring that still leaves m1536 as the sweet spot at current fidelity. So the most useful next step is improving the compensation quality at m1536 or m1280, and I'm considering whether more compensation data is the right lever.

Next I'm weighing whether to spend more passes on extra compensation data for layers 0 through 4, which would add about half again as much data per layer and might cut test error by roughly 5 percent. I'm also considering a downstream-aware objective, since the error in each hidden state only matters for the final logits along certain directions, though attention makes that map harder to pin down for the earlier layers.

I'm considering a correction at the end instead: fitting layer 5's FFN on the CLS position to match the base model's targets, using the edited model's CLS inputs from the fixed-size run, so the accumulated drift from layers 0 through 4 gets compensated through the kept units' output weights.

I'm checking whether the layer-5 CLS data from the 1024 run lines up with the base model's activations for the same inputs. If it does, I can measure how much of the gap to the base targets a corrective output-weight update on the kept units could recover.

I'm checking whether a correction on layer 5's FFN could recover the gap, but the earlier head refit gave no gain, which suggests the drift is random-like rather than linear. With only about 1000 samples and 1536 nonlinear features, I expect a CLS-only FFN correction to overfit, so I'm doubtful it would help much.

I'm now checking where the logit noise actually comes from: layer 5's attention over the layer-4 tokens and its FFN on the CLS position. Earlier results suggest layer 5 adds a lot of error at m1024, and the compensation for it was fit on only about 2000 CLS samples, which is thin for 1536 features. More layer-5 CLS data could be worth collecting, if it's cheap to get.

I'm going to check layer 5's compensation on the CLS data at several unit counts, using 5-fold CV split by id so base and edited copies of the same input never land on both sides. I'll measure the error in logit space, since that is what actually determines the answer.

I'm now checking whether the layer-5 FFN outputs at the CLS position can be evaluated for their effect on the logits, using the activations already read from the probe passes. The worry is that doing this partway through the last layer counts as re-running the forward pass, which I want to avoid, so I'd keep the calculation to closed-form math on observed activations.

I'm going to keep the calculation to closed-form math on activations I've already read, rather than re-running the model. Evaluating the layer-5 FFN output through the final norm and head on those observed activations is effectively just the fitting objective, so I'll measure the error in logit space that way. I'm also checking where the layer-5 CLS data came from, since it matters which pass produced it.

I'm tracing which probe pass produced each layer's data: passes 14/15 cover layer 0, 16/17 layer 1, 18/19 layer 2, 20/21 layer 3, and 22/23 layer 4. Pass 24 used six layers and read the layer-5 CLS states, so that's likely where the layer-5 compensation data came from.

I'm going to verify whether pass 24 really read the L5 layer output. If its CLS hidden state is just a layer-normalized input with no affine, the row mean should be near 0 and the variance near 1. If that holds, my earlier m1024 full-model agreement figure of .896 is invalid, and the m1024 full model may never have been evaluated. I'd also check whether pass 25 was the m1536 full-model evaluation.



The holdout gap is about 1.3–1.7× train error, so the fit looks somewhat data-limited and ridge strength barely matters. I also need to recheck whether pass 24 was actually the layer-5 probe, which would make my .896 agreement figure invalid.



## tool_use: run_python

```json
{
 "code": "h=np.load('pass_24/cls_hidden_L6.npy')\nprint(h.mean(1)[:5], h.var(1)[:5])\na5=np.load('ffn_ed_L5_cls.npy'); g=base('roberta.encoder.layer.5.attention.output.LayerNorm.weight'); b=base('roberta.encoder.layer.5.attention.output.LayerNorm.bias')\nprint(np.abs(h*g+b - a5).max())\nh9=np.load('pass_9/cls_hidden_L6.npy'); print(json.load(open('pass_9/meta.json'))['structure'] if 'structure' in json.load(open('pass_9/meta.json')) else json.load(open('pass_9/meta.json')).keys(), h9.mean(1)[:3], h9.var(1)[:3])\na5b=np.load('ffn_in_L5_cls.npy'); print(np.abs(h9*g+b-a5b).max())\n"
}
```


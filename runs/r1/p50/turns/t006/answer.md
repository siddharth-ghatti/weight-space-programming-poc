## reasoning summary

I'm comparing the in-sample accuracy of the base model against its test accuracy, and the gap suggests the in-sample numbers are inflated. The probes on the practice data are likely inflated too, since the representations were fine-tuned on those same examples. I'm now looking at how often the layer-5 and layer-4 probes agree with the base model on test, where the 5% disagreement is what I want to understand.

I'm thinking that test examples are out-of-distribution relative to the fine-tuning, so the probe fitted on in-sample representations probably agrees less with the base model on test than on practice. Since the base model is less confident on test, small perturbations would flip more predictions, which makes the 95% agreement risky. I'm also weighing an alternative that keeps all six layers and narrows the FFNs instead of dropping layers.

I'm weighing whether a low-margin, in-sample proxy can stand in for how the probes would behave on test, since the base model's confidence drops there. Next I need to run experiments to see which trade-off between dropping layers and narrowing the FFNs holds up better.

I'm weighing how robust the probes are by checking how much the logits shift on practice data relative to the margins, since I don't have test data to measure margins directly. I'll aim to keep perturbations small against that margin distribution. I'm also considering whether the last-layer practice representations cluster too tightly by class, which could make the test agreement depend on how the perturbation interacts with test examples. Option A is to keep all six layers and prune FFN units, which often allows roughly half the units to go with modest loss, using activation-based importance.

I'm considering dropping a middle layer instead of the last one, since the residual stream should mostly survive and the original head might still work. Before deciding, I need forward passes to check it. With a budget of 48, I plan to test each single-layer drop with the original head by running predictions on the 1000 practice examples and measuring agreement with the base model.

I'm weighing whether to use reads instead of plain predicts, since they return both the logits and the last-layer CLS state for refitting the head. For the middle-layer drops, I can't estimate the effect without a forward pass, and I'm relying on the general pattern that dropping early layers hurts badly while dropping middle ones is mild.

I'm thinking about zeroing out a layer's FFN and reading the hidden state at that layer's output, since layers before it stay untouched. Only one layer can be modified per pass, so covering all six layers takes six passes.

I'm weighing whether I can pull the FFN inputs out of the read activations without modifying the model. Recovering them from the layer output isn't straightforward, so I'm considering using the previous layer's output as a rough stand-in, though it won't match exactly. Before settling on that, I want to check whether this kind of activation-based unit importance is reasonable to compute from the reads.

I'm weighing whether computing FFN unit activations from the read outputs counts as running the model myself, which would be a grey area. Simpler alternatives would be weight-norm importance, or mean activation statistics, though those miss the nonlinear distribution of the activations.

I'm weighing whether to compute FFN unit activations from the read hidden states, which would be closed-form math on returned activations rather than a full re-simulation of the model. The harness doesn't expose intermediate activations directly, so the question is whether that derivation stays within the rules before I commit to it.

I'm weighing whether to derive FFN unit activations from the exact inputs returned by a single-layer edit pass, which would let me rank units and refit the output projection in closed form. I'd keep that math confined to one FFN sublayer and avoid using it to simulate whole-model outputs, and I'd note the approach in my notes. I'm also checking whether reading the layer output at each step, rather than running six separate passes, would still give the inputs I need.

I'm weighing whether to pull FFN unit activations out of a single-layer edit pass, since the layers after it stay untouched and the read at each layer's output gives me the exact input I need. Only one layer can be edited per pass, so this stays within the 48-pass budget, though I'd want enough tokens per example to rank all the units.

I'm now comparing parameter budgets for each layer count, with the FFN width at 512 or 1024. Dropping to 4 layers at width 512 lands near 52M, and I'm weighing that against the 6-layer options.

I'm now going through the parameter budget for each layer count, and the trade-off is clear: each layer removed saves about 2.4M plus its FFN, so dropping one is the cheapest way to shrink the model. My plan is to first test dropping layer 4 with the original head and refit, since layer 5's effect is already known from the base reads.

I'm planning to test dropping layer 4 next and check it with the original head before refitting. The plan is to run one pass per structure, since the structure file is read at forward-pass time, so each candidate has to be tested sequentially.

I'm thinking about getting the head to map the edited model's features onto the base logits. The idea is to read the pooled output from the edited model and fit out_proj by least squares, which stays within the returned activations. I could also make the dense layer nearly linear and rescale out_proj to get a direct linear map from the hidden state to the logits.

I'm going to try a direct linear map from the hidden state to the logits, either by setting the dense layer close to linear and rescaling out_proj, or by keeping the original dense and tanh and refitting only out_proj on the returned pooled outputs. I'll compare the two using cross-validation.

I'm going to run the pass with layer 4 dropped, keeping layers 0 through 3 and 5, and read back the last-layer CLS state, the pooled output, and the logits.



Base in-sample accuracy is 92.5% versus 77.5% on test, so the practice set is heavily overfit and I'll judge edits by agreement with the base predictions. A refit on the layer-5 state reaches about 95% agreement and layer 4 about 91%, so cutting top layers loses fidelity fast; next I'm testing dropping a middle layer, layer 4, while keeping the original head.



## tool_use: run_python

```json
{
 "code": "print(set_structure(layers=[0,1,2,3,5], ffn_keep=3072))\nprint(structure()['layers'])"
}
```


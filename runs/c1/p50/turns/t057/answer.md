## reasoning summary

I'm running an experiment on dropping layers, working from a structure like the eleven-layer list minus one index. Each dropped layer saves about 7.09M parameters, while halving the FFN pruning saves about 2.36M per layer, roughly 26M in total, which is about the same as removing 3.7 layers. I'm starting to set up the first pass family, D1, with a layer structure that skips a few of the middle indices.

I'm settling on dropping layers 4 and 6, which gives the structure [0,1,2,3,5,7,8,9,10]. Adjacent drops look worse than spaced ones, since the middle layers are the most redundant. I'll read mean_hidden at every layer for the probe ids, and I'll also keep in mind that dropping layer 6 removes a carry head.

I'm weighing a pruning approach that doesn't need per-token FFN inputs, since reading those would take a separate pass for each of the ten layers. I'm also checking whether the FFN scores were computed in a way I can reuse, and whether a layer-norm trick could expose FFN inputs for several layers at once.

I'm working out how to get FFN inputs without a separate pass per layer, since dropping the rest of the FFN would break downstream layers. Reading the layer-norm output for one layer per pass still covers 64 ids, about 1,900 tokens, which might be enough for selection if the least-squares refit is regularized.

I'm checking how the FFN scores were computed, and whether the standard deviations line up with activations I could reproduce from the layer inputs. I don't remember the details, so I'll test it directly. My plan is to run one layer-dropping pass next to see how the structure behaves.

I'm weighing whether to pick units by activation energy weighted by their output weights, then refit the output projection by least squares over the kept units. That would need per-token activations at each layer, which the layer-norm trick can provide one layer per pass.

I'm checking whether I can pull FFN inputs for several layers in a single pass, but exposing one layer's output breaks the inputs to every layer after it. Reading the FFN input from one layer per pass seems unavoidable, so the pruning step means about ten passes. With evaluation passes, that leaves roughly 29 for the layer-drop and classifier work.

I'm checking whether the content stack can stop earlier: reading from the output of layer 8 instead of layer 9 gives about 98.4% alone and 98.5% combined with the input, so ten content layers would be enough. At a 1024-unit FFN that comes to roughly 63.8M parameters, and 512 units gets it to about 56M.

Next I'll run the D1 layer-drop pass, dropping layers 4 and 6 to keep indices 0,1,2,3,5,7,8,9,10,11. I'll compare the depth-matched hidden states against the base model, where the output after layer 8 is about 98.4%.

I'm setting the structure to keep layers 0,1,2,3,5,7,8,9,10,11 with 3072 FFN units. The current edits still carry bias compensation from the earlier pruning run on layers 0 through 10, so I need to drop those before the pass. I'll leave the layer 11 edits in place, since they don't affect the intermediate means I'm reading here.



I'm dropping layers 4 and 6 in D1 at full FFN width to test layer dropping, then comparing each depth's features against the base. First I'm removing the leftover exp-A bias edits on layers 0–10.



## tool_use: run_python

```json
{
 "code": "for l in range(11):\n    drop_edit(f'bert.encoder.layer.{l}.output.dense.bias')\nprint(edits())\nprint(set_structure(layers=[0,1,2,3,5,7,8,9,10,11], ffn_keep=3072))\n"
}
```


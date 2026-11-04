## reasoning summary

I'm finding that the energy is spread across many units, with a few layers having dominant outliers. Since the FFN energy is fairly evenly distributed, narrowing it may hurt representation fidelity, though the classification features might still hold up. Experiments are costly, so I want to pick the most efficient design. My first plan is to structure layers 0 through 10 at width 1024, ranking units by std score and adding bias compensation.

I'm laying out the experiments now: a structure over layers 0 through 10 at width 1024, reading mean and CLS features at every layer, so one pass gives the depth curve for that width. A second run at a width near 384 would show how far the degradation goes, and a layer-skipping variant could test whether dropping some depths costs less. For compensation, I'd fold the mean contribution of the removed units into the output bias, though the proxy activations used for those means may be off.

I'm weighing whether bias compensation is worth doing if the proxy activations are off, though the LN after the FFN should make the mean shift matter less, so I'll go ahead with it. Next I'm planning a helper that builds edits for a given layer set and width, selecting the top units by std score per layer and saving the output bias adjustments for the kept layers.

I'm weighing whether the final pooling layer should be original layer 11, since its weights get fully rewritten anyway and it's already set up for pooling. That only works if its id sits after the last kept real layer. I'm also wondering whether the last real layer could take on the pooling role itself, so I'd skip a separate layer.

I'm weighing whether the last real layer could absorb the pooling role, so I wouldn't need a separate layer. One attention head could be turned into a carry head that writes ridge scores of the mean hidden state into the reserved dimensions, while the other heads still produce the CLS output. The pooler would then be a ridge on that CLS vector plus the decoded reserved dimensions.

I'm comparing a design where the last real layer absorbs the pooling role against one with a separate pooling layer. The idea is to take the ridge scores of the mean hidden states from several layers, carry them into reserved dimensions through a few heads, and let a linear pooler decode them alongside the CLS output. A layer-10 setup with four features already reached about 97.7% in the linear fit, so I'm checking whether that holds without the extra layer.

I'm now weighing a nonlinear pooler: random features over the final CLS vector, including the carried scores, though those scores are supervised and would need nested fitting for a fair CV estimate. The harder issue is that the carried scores are fit jointly with the final ridge, which makes the whole thing a single linear model.

I'm weighing whether the carry heads can be installed with placeholder value and output weights, since the reserved dimensions are isolated from the CLS features and so the features read beforehand stay valid. The catch is that swapping a normal head for a carry head changes the representation, so the features will shift once the edit is in place.

I'm reusing the carry heads already installed in layers 2, 6, and 10, so I only need to refit their value and output weights. Reserved dimensions are isolated from the other features, which means I can set those weights after reading the features without changing them. I'm planning to keep layers 0 through 10 for this design.

I'm sketching a design that keeps layers 0 through 10 and drops layer 11, with carry heads in layers 2, 6, and 10 writing mean-hidden-state scores into the reserved dimensions. The final CLS from layer 10 would feed a ridge pooler, and a linear fit on those features should land near 97.7%. That only trims about one layer's worth of parameters, so it doesn't reach the size target yet.

I'm weighing a leaner setup: keep the FFN at a much smaller width, or drop it entirely, to see whether the mean features still classify well. Then I'll run three checks on the layer 0 through 10 structure at widths 1024, 256, and 0, starting by confirming whether set_structure accepts a width of zero.



FFN importance is spread across most layers, so narrowing will distort representations, though the classifier is refit regardless. Next I'll check whether the harness accepts very small m, including zero.



## tool_use: run_python

```json
{
 "code": "for m in [0,1,8]:\n    try:\n        print(m, set_structure(layers=list(range(11)), ffn_keep=m))\n    except Exception as e: print(m,'ERR',e)\nclear_structure(); print(structure()['m'])"
}
```


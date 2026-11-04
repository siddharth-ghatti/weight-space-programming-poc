## reasoning summary

I'm weighing whether to drop some layers to make the model cheaper, which would mean keeping a subset like layers 0, 1, 8 plus pooling. The worry is that BERT's residual stream may not tolerate skipping middle layers well, since post-norm layer normalization could shift the activation distributions.

I'm working out the parameter budget for the FFN-pruning option: ten layers at m=1 come to about 48.1M, against 45.7M for three full layers. I'm also questioning whether attention or the FFN matters more for feature quality, since dropping the FFN may break how later layers attend. The budget equivalences make it clear that one full layer equals roughly three layers at m=1.

I'm planning how to split the 23 passes: roughly ten for exploring structures, six for building the final model, and the rest for validation. For FFN pruning, I'm considering that a single read of an edited model gives per-layer means I can evaluate across prefix combinations at once. I'd pick FFN units by activation importance, but the post-attention input isn't directly exposed by the token read, so that would need recomputing.

I'm weighing whether mean-folding can compensate for pruned FFN units, since the expected GELU activation of each unit isn't available without the activations themselves. Without that term, removing a unit shifts the FFN output mean, and I'm considering how much that matters given how sparse the activations are in BERT.

I'm checking whether the mean of each FFN unit's pre-activation can be estimated from the weights alone, since its input is the post-attention LayerNorm output with gain and bias. That gives a rough mean from the bias and the β term, but the spread is unreliable because BERT's hidden states are anisotropic with outlier dimensions. I'm also considering whether a different design, one that doesn't keep BERT's deep layers as they are, would avoid this estimation problem.

I'm checking the kernel results for the layer subsets, where [1,9] stands out at 98.1%, well above the others. Single layer 9 is also decent on the kernel. I'll set aside the CLS readout at layer 12, since it doesn't help here. The idea of skipping layers 2 through 8 is tempting, but feeding layer 9 the output of layer 2 instead of layer 9 probably means it works on an input it wasn't trained for.

I'm weighing whether layer 9 could serve as the pooling layer, which would make the stack 0 through 8 plus pooling, ten layers total. At full width that comes to about 95M parameters, which is too expensive, so the real question is whether pruning the FFN width uniformly across many layers keeps the features usable.

I'm sketching two more experiments: all 12 layers at m=1 for an attention-only read, and a uniform m=512 pruning chosen by a weight heuristic, to see how sensitive the features are. Counting parameters, a two-layer version with full FFN lands around 38.6M, and three layers with the same width reach about 45.7M, but neither looks accurate enough. That points toward keeping many attention layers with a thinner FFN.

I'm going to run the experiment with all 12 layers at m=1 to see how an attention-heavy structure holds up. Removing the FFN leaves only its output bias, so the mean FFN contribution is lost, and I'm wondering whether that can be folded back in if I can estimate it.

I'm settling on one run with all 12 layers at m=1 and reading mean_hidden plus the CLS hidden state for 1024 ids, which keeps the noise near half a percent. Then I'll cross-validate the per-layer means and their combinations.

I'm thinking about whether a single run with all twelve layers at m=1 would show how far attention-heavy features hold up, and whether the minimum-norm output column makes that run cleaner. I'm also weighing an alternative: distilling the deep features into a shallow model in closed form.

Now I'm working out a cleaner way to fold the pruned FFN's mean back in, and I'm weighing whether a closed-form estimate from the layer-norm gain and bias could stand in for the activations I can't read directly.

I'm weighing whether a uniform-width pruning across all twelve layers, at m=1, can be read in one pass and tested on several layer combinations at once.

Next I'm setting up the attention-only network with all twelve layers, keeping in each layer the FFN unit with the smallest output-weight norm.



Pairs with a deep layer reach 98.1% with a kernel readout, but a 10-layer stack is too costly. I'm now testing a 12-layer attention-only stack with the FFN reduced to a bias, to see whether depth survives thin FFNs.



## tool_use: run_python

```json
{
 "code": "keepu={}\nfor L in range(12):\n    Wo=W(f'bert.encoder.layer.{L}.output.dense.weight')\n    keepu[L]=[int(np.argmin(np.linalg.norm(Wo,axis=0)))]\nprint(keepu)\nprint(set_structure(layers=list(range(12)), ffn_keep=keepu))\nwith open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- base reads (pass1,2) -> base_reads.npz (mean/cls all 13 layers, all 1496). reserved dims in reserved.npy (exclude from CV: leaked).\n- CV linear ridge per layer mean ~95.5-96.5%; L0 (embeddings) 96.0%. Prefix all-means linear: j=9 97.5, j=12 97.9.\n- RBF kernel ridge (gamma .3-.4 on z-scored/768, lam .003): M1 97.2, [1,6] 97.55, [1,9] 98.1, [2,6,10,11] 98.0. Need depth.\n- Exp A (pass 3): all 12 layers m=1 (min-norm unit) to test attention-only features.\n\"\"\")\n"
}
```


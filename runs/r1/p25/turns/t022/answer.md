## reasoning summary

Truncating to 5 layers still beats refitting a 1-layer drop: the refit only reaches about 94.4% agreement against 95.5%. That puts the ceiling near 95% for any single-layer drop, which may not be enough. Since substantive compression lands around 93-95% agreement in practice, I'm now checking whether the test b

I'm weighing whether a richer refit could recover the gap, since a linear head on the final CLS may be leaving accuracy on the table. Fitting out_proj on the head's tanh features, or ridge-fitting the pre-tanh 768-dim values and reapplying the base tanh and out_proj, both have closed-form solutions, so I plan to test them on the truncation-5 features where the pass-1 data is free to use.

Now I'm weighing whether the best low-compression option is pruning layer 5's feed-forward block, measured only on the CLS position since that's the only position the final layer's output depends on.

The shared head size means layer 5's FFN can't be pruned harder than the others, so the trimming has to pick which units to keep in each layer. The attention side of the final layer has nothing to save, since only the CLS query matters there anyway. Total cost is roughly 14.2M plus 9222 times m across six layers, so the real question is which per-layer unit choices hold up on CLS.

I'm considering whether merging layers could work: folding layer 5's feed-forward units into layer 4's, which would mean the combined width has to stay within the 3072 units each layer has, so I'd need to see whether that limit holds up.

Keeping the search on pruning and merging options that might hold 98-99% agreement, since a shared width m across all layers caps how much I can trim per layer. Merging layer 5's FFN into layer 4's doesn't fit within the 3072-unit limit, and dropping layer 5 outright loses too much CLS-side attention work. I'm weighing whether a per-unit importance measure on the CLS position alone can justify which units to keep.

I'm looking at whether the test set might be far less forgiving than the practice agreement suggests, since low-margin items drop to about 82% under truncation to 5 layers, which puts test agreement near 88% and likely a failing result. That pushes me toward needing near-perfect function preservation, so I'm evaluating whether mild FFN pruning, backed by exact FFN-input statistics from probe passes, can hold that level. I also want to check whether an exact recovery trick would be permitted in the edited model.

I'm working out the closed-form recovery: with the FFN output zeroed and LN2 set to identity, the layer's output is just a normalized version of the attention-plus-residual value, and LN1's gamma and beta give constraints I can solve for the per-token scale and offset.

I'm checking whether the LN1 constraints can pin down each token's scale and offset, since the mean-zero and unit-variance conditions give a closed-form solution for the pre-normalization vector. Working through it, the offset comes out linear in the scale, and the variance condition reduces to a quadratic I can solve, keeping the positive root.

I'm checking whether the Gaussian approximation is what limits the FFN-input statistics, since the exact recovery would give true inputs for each layer in about one pass each. Layer 0's proxy looks like the weakest, while layers 1 through 4 are roughly usable at 0.7 to 0.8 overlap.

I'm asking whether computing the FFN unit activations directly on the returned layer inputs counts as allowed closed-form math, since it's a single layer's computation rather than a simulated forward pass. The rule I'm weighing bars re-implementing the model to evaluate pruned configurations without spending passes, so the distinction matters for whether this route is permitted.

I'm wondering whether the forward pass could hand back the unit activations directly, perhaps by zeroing the attention and LayerNorm parameters so the FFN input becomes trivial. That runs into a problem, since the FFN input is the attention output itself and can't be made zero without losing the signal I need.

I'm weighing whether a diagnostic run without bias compensation, or with compensation built only from the input proxy, would show how much of the fidelity loss comes from the FFN pruning itself rather than from the approximations, though each variant costs a pass.

Budget is tight at 21 passes, so I'm weighing six probe passes to capture true FFN inputs for every layer, reusing the same 32 ids so the true outputs come along. A cheaper idea is to probe several layers in one pass, but that only works if the edited model can chain kept layers, so I'm checking whether layer l can be isolated that way.

I'm weighing whether to probe the true FFN inputs for each layer, since the best option I have is six extra passes. Layer 0 is the weakest proxy, so it matters most. Dropping a layer entirely stays low-fidelity around 95%, which leaves six layers as the likely answer.

Sizing the budget, I'm comparing parameter counts for a 6-layer model at widths 2560, 2048, and 1536, which save about 5.7%, 11.5%, and 17%. The gains look modest, so I'm wondering whether a linear stand-in for each pruned unit could reach a width near 1024 at high fidelity, which would save about 23%.

I'm now working out how to build the pruned FFN as a linear stand-in: collapse the pruned units into one matrix M, then take its top singular directions as a small set of units. I'll keep their biases high enough that every token stays in the linear region of GELU, which means shifting the bias by several standard deviations of the input projection.

I'm checking whether the linear stand-in can hold up: a scaled input keeps the GELU in its near-linear range, and the bias shift keeps every token well inside it. The open question is how much of the FFN variance that linear part actually captures.

I'm settling on exact FFN input statistics: a probe pass on the same 32 ids as before lets me pair true inputs with outputs. Then I'd pick about 900 nonlinear units and about 124 linear directions from the top of M's spectrum to form the width-1024 layer.

I'm weighing whether the CLS token needs its own importance measure, since its FFN inputs look very different from the other positions and may be dominated by a few units. Because the last layer's output only depends on CLS, I'd also need CLS-specific statistics for that layer.

I'm considering whether probing the CLS token's FFN inputs separately is worth it, since reading cls_hidden can cover 1000 ids while token_hidden is capped at 32. The question is whether the CLS statistics that matter for the final classification can be gathered without mixing the two calls.

I'm planning to run the probe passes on the same 32 ids as before, so the captured FFN inputs line up with the base hidden states. Each probe keeps layers up to l intact and zeroes out layer l's FFN and normalization, so the edited output exposes the FFN input directly.

I'm checking the layer-0 probe setup: each pass zeroes layer l's attention output and second normalization so the edited output exposes the FFN input directly, and I'm keeping only layers 0 through 5 for six passes total, which leaves fifteen.

I'm weighing whether to test m=2048 with the current proxies first, since one pass could show if fidelity is already near 98%. If it is, better statistics might push m=1536 to that level, but the proxy overlap of 0.74 to 0.84 suggests the averaged importance is only about 85% right, so I'm leaning toward investing in the true FFN input statistics anyway.

I'm going to probe the true FFN inputs for each layer next, reusing the same 32 ids so they line up with the base hidden states. The probe setup is what I'm sketching out now.



Layer-3 diagonal compensation scores 94.4% after head refit, below plain top-layer truncation at 95.5%, so single-layer drops top out near 95%. Next I'll set up a probe that reads the true FFN input for layer 0.



## tool_use: run_python

```json
{
 "code": "for e in edits(): drop_edit(e)\ndef setup_probe(l):\n    for e in edits(): drop_edit(e)\n    pre=f'roberta.encoder.layer.{l}.'\n    save_edit(pre+'attention.output.LayerNorm.weight',np.ones(768,np.float32))\n    save_edit(pre+'attention.output.LayerNorm.bias',np.zeros(768,np.float32))\n    save_edit(pre+'output.dense.weight',np.zeros((768,3072),np.float32))\n    save_edit(pre+'output.dense.bias',np.zeros(768,np.float32))\n    save_edit(pre+'output.LayerNorm.weight',np.ones(768,np.float32))\n    save_edit(pre+'output.LayerNorm.bias',np.zeros(768,np.float32))\n    print(set_structure(layers=list(range(l+1)),ffn_keep=3072))\nopen(WORK/'probe.py','w').write('''\nimport numpy as np\n''')\nsetup_probe(0); print(edits())\nwith open(WORK/'notes.md','a') as f: f.write('''\n## Results so far\n- pass1: base read all 1000 (cls/mean all layers, pooled, logits). base practice acc 0.925.\n- pass2: base token_hidden 32 ids (tok_ids in pass_2/meta.json), stats tok_mu.npy/tok_cov.npy.\n- Probes: truncate to 5 layers + ridge refit to base logits -> agree 0.955, low-margin agree 0.82 (too lossy?)\n- pass3 exp A: 6L m=1536 proxy importance + bias comp: orig head agree 0.917, refit 0.93. Bad (layer0 proxy poor).\n- pass4 exp B: drop layer3 + diag LN comp: orig head agree 0.59, refit 0.944. Worse than truncation.\n- Plan: probe true FFN inputs per layer (LN1->identity, FFN zeroed, LN2 identity => read n_l; a=g1*n+b1). passes 5-10.\n- Idea: 'linear units' (large bias, gelu linear regime) implement low-rank linear part of pruned units.\n''')"
}
```


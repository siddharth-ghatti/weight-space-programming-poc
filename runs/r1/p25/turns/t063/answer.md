## reasoning summary

The error looks evenly spread across the output dimensions with none dominating, and since these numbers are in-sample they're a bit optimistic compared to CV. Weighting the selection score by dimension might help slightly, but probably not much. The bigger levers seem to be more data and sequential correction, or just accepting m=2048. Let me revisit the cost trade-off.

I'm weighing whether the scoring rewards a small parameter cut enough to justify the failure risk. Dropping from 72.86M to 68.14M is about a 6.5% gain. My plan is to try m=1536 with least squares first, and if agreement and low-margin both clear their thresholds, add sequential correction and more data to push accuracy higher, otherwise fall back to m=2048 or 1792. I'm also checking whether sequential correction for layer 5 can ride along on the same measurement pass, since that would need the edited model's layer-5 FFN input at CLS across all 1000 ids.

I'm weighing whether to bolt sequential correction onto the m=1536 measurement pass, since the probe would give me layer 5's edited inputs for all 1000 ids. I'd still rely on real forward passes for evaluation rather than recomputing the final logits myself, but fitting the layer-5 FFN against FFN targets in closed form is ordinary least squares on activations I already have.

I'm planning a two-pass sequential refit at m=1536, starting with a probe of layer 1's inputs on the edited model, since that input depends only on layer 0's edit. I'm also reconsidering the target for that refit, checking whether the base's pre-LN2 vector, a plus f applied to a, is the right thing to match against the edited model's output.

I'm working out the sequential refit cost: about five passes for layers 1 through 5, each waiting on the previous one, plus a final check. That totals roughly 6 to 7 of the 13 passes, plus one for the m=1536 measurement, leaving a few spare for extra data. Whether the FFN can learn much of the systematic gap is uncertain, so cross-validation will decide.

I'm weighing a cheaper option: correct only the last layer, since its CLS output has the most direct effect on the logits and 1000 samples is enough for a reliable least-squares fit. The worry is that errors in layer 5's CLS input come from attention over corrupted tokens, so the FFN can only partly compensate. I'm also considering whether one probe pass with layer 5 set up as a placeholder could collect the m=1536 measurements and the layer-5 inputs together.

I'm going to run the m=1536 least-squares edit first, with layer 5 in probe mode, so one pass returns the cls and mean hidden states for every layer on all 1000 ids. Since fixing upstream layers changes what layer 5 sees, layers 1 through 4 have to be corrected before the layer-5 refit, and I still don't know whether m=1536 clears the bar.

I'm planning the layer-5 probe in pass 13 so it returns the upstream states as well, and I'd fit layer 5 against the base's CLS target, x_base minus the edited input. Then I'd check agreement against the m=2048 baseline of 0.985 before committing to the sequential corrections for layers 1 through 4, which would take about eight passes in total.

I'm settling on a per-layer probe for layer 5 that sets its LayerNorm and FFN to identity-like values, so one pass returns its inputs alongside the upstream states. The fit then uses ridge least squares on 1000 samples across 1536 features, with λ picked by cross-validation.

I'm setting layer 5's LayerNorms and output projection to neutral values for a probe pass, so I can read its inputs off the edited model. I'll check that its CLS input matches what I computed earlier from the base, then build the m=1536 least-squares edit for layers 0 through 4.

I'm setting up the layer-5 probe alongside the m=1536 structure, so that layers 0 through 5 are built with 1536 units while layer 5 returns its inputs instead of producing a real output.



Dimension weighting won't fix the spread-out error, so I'm moving to more probe data and sequential refitting; next I'll build the m=1536 edit for layers 0–4 and run layer 5 in probe mode to collect its inputs for refitting.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('lsbuild.py').read())\nkeep=apply_ls(1536, base, W, save_edit, layers=range(5), lam=100.0)\npre='roberta.encoder.layer.5.'\nsave_edit(pre+'attention.output.LayerNorm.weight',np.ones(768,np.float32))\nsave_edit(pre+'attention.output.LayerNorm.bias',np.zeros(768,np.float32))\nsave_edit(pre+'output.dense.weight',np.zeros((768,3072),np.float32))\nsave_edit(pre+'output.dense.bias',np.zeros(768,np.float32))\nsave_edit(pre+'output.LayerNorm.weight',np.ones(768,np.float32))\nsave_edit(pre+'output.LayerNorm.bias',np.zeros(768,np.float32))\nkeep[5]=list(range(1536))\nimport json; json.dump({str(k):v for k,v in keep.items()},open('keep_1536_L0-4.json','w'))\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep=keep))\ns=open('notes.md').read()\ns+='''- pass12 RESULT m=2048 LS: orig head agree 0.985, low-agree 0.94, logitR2 0.996, acc .916. pairwise logit pert median .147 p90 .436 (pass11: .73/1.84). <-- fallback candidate (rebuild: apply_ls(2048,...) all layers, default lam).\n- m=1536 CV errs: L0 .091 L1 .132 L2 .068 L3 .011 L4 .004 (lam100 best).\n- pass13 planned: m=1536 LS L0-4 (lam100) + layer5 PROBE mode (LN1/LN2 identity, FFN zero) -> cls_hidden_L6 = z5 edited (pre-LN1) for 1000 ids; then refit L5 to target x_base - a5_e (sequential correction).\n'''\nopen('notes.md','w').write(s)\n"
}
```


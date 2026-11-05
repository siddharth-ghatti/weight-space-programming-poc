## reasoning summary

I'm checking how concentrated the energy really is: the top 1024 values hold roughly 68 to 84 percent, and the top 1536 hold 80 to 90 percent. Pruning the FFN to half would keep about 85 percent of the energy, which leaves a relative error near 0.39 if the errors add incoherently, and that seems significant. I'm also noting that this weights-only estimate is rough.

I'm weighing whether to keep the FFN width at 1536 or go to 2048 for more margin. The trade-off is parameter count against feature quality: a larger FFN costs more per layer, but the residual stream and attention should soften the perturbation from pruning.

I'm weighing whether a width of 1536 is a good balance, since the features stay fairly faithful there. The bigger cost seems to come from losing depth rather than FFN width, because the refit head can compensate for the width. I'll check which combination of layers and width gets the smallest model that still passes cross-validation.

I'm now weighing whether a middle width like 1536 is enough, or whether 2048 is worth the extra cost. I could check the 6-layer pruned model against the base model's in-sample accuracy at that width to see how much pruning actually hurts.

I'm weighing whether to add a bias compensation step to the FFN output, which would fold the pruned units' average contribution into the output bias. I'll work out that average under a Gaussian assumption to see if it's worth doing.

I'm working out whether pruned FFN units would shift the output in a systematic way, by estimating the average contribution of the dropped units under a Gaussian assumption. If that shift turns out to be tiny compared with the kept output, I'll skip the bias correction.

I'm weighing whether to add a bias correction for the pruned units, estimating the mean shift they'd leave behind. The shift looks modest, but since it would be the same across all tokens, it might push the hidden states in a consistent direction, possibly along the outlier dimensions, so I'm unsure whether it's worth doing without real activation data.

I'm settling on skipping the bias compensation, since I can't verify the shift without real activations. For the selection, I'll score each FFN unit by its typical output magnitude, and I'm adding "pooled" to the read's outputs alongside the hidden states and logits so I can check my offline head.

I'm weighing the FFN width against the number of layers: m=1536 looks like a balanced choice, with the parameter counts for keeping one through four layers working out to roughly 44 to 54 million.

I'm settling on an FFN width of 1536, since 1024 would save only a few million parameters and risks hurting every layer. Next I'll set up the pruned structure, confirm the parameter count, and run the read on the edited model.



The weight-only estimate says the top 1024 FFN units keep only about 68–84% of output energy, which could hurt the features and can't be undone after one read. I'm reading with m=1536 (half the FFN) on all 6 layers, picking units per layer by estimated activation energy times output-column norm, with no other edits.



## tool_use: run_python

```json
{
 "code": "\nimport numpy as np, json\nimp = np.load('imp_gauss.npy')\nm=1536\nkeep = {l: sorted(np.argsort(-imp[l])[:m].tolist()) for l in range(6)}\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep=keep))\ns = structure(); print(s['layers'], s['m'], edits())\nopen('notes.md','a').write(\"\\n- Read structure: layers 0..5, m=1536 per layer, top units by gaussian-energy*Wout-norm (imp_gauss.npy). No weight edits yet. About to run pass 2 (read edited, all 1000 ids).\\n\")\n"
}
```


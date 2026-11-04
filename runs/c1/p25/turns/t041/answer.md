## reasoning summary

I'm reorienting on the experiment log: the last forward-pass result is pass_7, Exp D, but the notes call for Exp E as pass 8, which would drop only layer 4. I count 18 passes left of 25, so passes 1 through 7 are used and pass 8 hasn't been run yet, and I'm checking that index against the notes.

The notes say Exp E (pass 8, dropping only layer 4) hasn't been run yet, since 18 of 25 passes remain and passes 1 through 7 are used. Next I'll check the working folder state, then weigh the strategy: the target is 97.8% test accuracy against a current 98.31%, and the earlier estimate for the start model was about 97.8%. My notes put a realistic kernel readout around 98.0%, which would still clear the bar with room to spare.

I'm working through a design for the readout: a 1496-unit FFN whose GELU features sit on the CLS vector, which holds the mean of layer-8 tokens. The catch is the LayerNorm stages around that FFN, which add nonlinear normalization before the pooler and classifier, so I'm considering whether the FFN output can be kept small against the residual so the normalization stays close to linear.

I'm weighing how few layers can be kept: the notes tie quality to layer count, with the 8-layer version around 97.55% and the 9-layer one around 98.0%. The bar is 97.8% on test, and the start model's test score ran about 0.5 points above its cross-validation figure, which may just be noise. Given how strict the criterion is, a 97.55% CV model could land anywhere near the threshold, so the risk is real.

I'm weighing whether to shrink the width further: the notes say cutting the FFN from 3072 to 1536 drops accuracy from about 97.8 to 97.1, so I'd keep 3072 and cut layers instead. Ten layers at that width is about 95.3M parameters, and dropping layer 4 as Exp E proposes would test whether the middle layers can be removed without losing much.

I'm checking whether dropping layer 4 but keeping layer 9 yields better features than keeping layers 0 through 8, since both options use the same number of layers. The idea is that if the 0-3, 5-9 stack matches the full model's quality, a layer could be removed for free.

I'm checking whether the edited model, with layer 4 dropped, reaches the quality of the base model with one fewer layer. If it does, that layer is effectively free to remove. I also need to remember that the reserved dimensions stay out of the cross-validation numbers.

I'm weighing a shallower stack against the test threshold: a 9-layer model at full width sits near 97.55% CV, which is a real risk against the 97.8% bar, while the 10-layer version lands around 98.0%. Narrowing the FFN width to about 2560 across 10 layers would cost roughly the same parameters as dropping one layer at full width, so the trade looks similar either way.

I'm considering whether the FFN pruning could be done better than the bias-folding approach, perhaps by least-squares reconstruction of the dropped units from the kept ones. That would need token-level activations, though, and computing them outside the model is off the table, so I'll set this option aside.

Given the remaining budget, I'd rather settle on something robust than keep chasing marginal size gains. Next I'll look at the working folder and then run Exp E as already planned.



## message

Picking up from the handoff: Exp E (drop layer 4 only) is set up in structure.json but its pass hasn't run yet (7 passes used). First I'll check the working folder so I know which files and analysis scripts exist.

## tool_use: run_python

```json
{
 "code": "import os, json\nprint(sorted(os.listdir(WORK)))\nfor d in pass_dirs(): print(d, sorted(os.listdir(d))[:6], len(os.listdir(d)))\nprint(structure()['layers'], structure()['m'])\nprint(edits())"
}
```


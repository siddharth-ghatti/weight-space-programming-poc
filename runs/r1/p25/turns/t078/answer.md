## reasoning summary

The m=1536 candidate C comes out at 0.978 agreement and 0.912 low-agreement, with logit R² of 0.995, while m=2048 stands at 0.985 agreement with 15 flips, against 22 flips here. The layer-6 CLS fit is 0.977 versus 0.979, but the layer-5 mixed fit seems to make up for it, leaving final CLS R² close to the m=2048 result. Still, the agreement gap is what I'd weigh.

I'm weighing a few cheap follow-ups now: a head refit to the base logits, which gives a small gain in cross-validation, and a second layer-5 correction pass that would need a fresh probe, with only a modest expected gain since that activation barely moved. I'm also checking whether the practice accuracy drop of about 1.7 points holds up on the test split.

I'm considering a bias correction: shifting each layer's LN2 bias by the average hidden-state deviation between edited and base models, since a constant offset in the output would be absorbed by the shift. I'd also check how much of the error a constant shift actually explains, and whether a per-dimension affine correction (diagonal gain and bias) would do better than that.

I'm thinking about getting the edited model's token-level hidden states for all layers on the same 32 probe ids, which would let me fit per-layer diagonal affine corrections on LN2 against the base. The catch is that correcting one layer shifts the inputs to the next, so each layer's fix from a single pass would only be approximate.

I'm weighing whether a systematic bias correction helps. Since the LS fit already compensates for the mean, a constant shift should be roughly zero on training tokens, so any remaining error comes from nonlinear propagation. I'm considering whether refitting the layers on edited token-level inputs in a single pass could handle that without double-counting the corrections.

I'm checking whether the attention output can be rebuilt from the attention probabilities and the layer inputs, since that would give the FFN input without another probe pass.

I'm weighing whether to accept candidate C at m=1536, with its 0.978 agreement, or go for a larger model. The m=2048 rebuild with new data would reach about 0.987 agreement at 72.9M parameters, and m=1792 about 0.982 at 70.5M, so the trade-off is agreement against size.

I'm weighing the flip counts more concretely now. C has 22 flips on practice versus 15 for m=2048, and since the test split is harder, the gap could widen there. The net accuracy hit is the part that worries me: most of those 22 flips go from right to wrong, so I want to see whether they cluster among low-margin items where the base model was already shaky.

I'm estimating how the flips would play out on the test split: if base is right on about half the low-margin flipped items and the alternative class is right on about 30%, each flip nets about -0.2, which would put the 22-flip candidate near -1.1 points. I'm not sure the test flip rate is really 2.5 times the practice rate, though, since I don't have the test margin distribution yet.

I'm estimating how flip probability depends on margin for candidate C, and considering whether test margins run lower than practice, since the base model is overfit (92.5% on train versus 77.5% on test). I'm also weighing whether to use the practice accuracy by margin bin, which is 70.5% under 1, 88% for 1 to 2, 97% for 2 to 4, and 99.3% for 4 to 8, as the basis for that estimate.

I'm estimating how many of the flipped test items would be low-margin ones, since the base model's second choice is probably not that reliable there. Overfitting means some high-margin test items could also be wrong, so the test margin distribution is probably skewed toward lower values, with perhaps two to two-and-a-half times as many items under margin 1 as on practice. I want the per-flip net effect on those low-margin items before I trust the rough -1.1 point estimate.

I'm working out the flip economics now: for practice items with margin under 1, I need the top-1 and top-2 hit rates, since each flip nets roughly the top-2 hit rate minus the top-1 hit rate. If the test split has about 2.5 times as many flips, the m=2048 candidate could still lose ground, which would make even that choice look risky rather than safe.

Checking the practice numbers directly now: the m=2048 model drops to 0.916 from 0.925 with 15 flips, which is a point of slack I don't want to spend. I'm going to pull the top-1 and top-2 hit rates for the low-margin bins so I can see whether a flip really nets negative.



Candidate C at m=1536 agrees 0.978 overall and 0.912 on low-margin items, but practice accuracy fell from 0.925 to 0.908, which worries me given the 1-point test margin. I'm now measuring how often flips hit the base's first versus second choice on low-margin practice items.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nlgb=np.load('pass_1/logits.npy'); y=np.load('y.npy')\no=np.argsort(-lgb,1); top1=o[:,0]; top2=o[:,1]\nsrt=np.sort(lgb,1); marg=srt[:,-1]-srt[:,-2]\nfor lo,hi in [(0,0.5),(0.5,1),(1,2),(2,4),(4,9)]:\n    s=(marg>=lo)&(marg<hi)\n    print(f'margin {lo}-{hi}: n={s.sum()} P(top1)={np.mean(top1[s]==y[s]):.3f} P(top2)={np.mean(top2[s]==y[s]):.3f}')\nfor pd in ['pass_12','pass_17']:\n    p=np.load(f'{pd}/logits.npy').argmax(1); f=p!=top1\n    print(pd,'flips',f.sum(),'base right',np.sum(f&(top1==y)),'new right',np.sum(f&(p==y)),'flip margins',np.round(np.sort(marg[f]),2))\n"
}
```


## reasoning summary

I'm weighing whether a larger regularization value might suit the smaller set of 752 anchors, since the smaller values clearly hurt results and seem to overfit. I'll try 752 anchors with lambda at 0.03 and 0.05, and keep the fit jitter in mind.

I'm considering whether a fundamentally better option exists, since the pruned model with K9 plus linear M0 already lands near base-level accuracy. Dropping a middle layer instead of the last one might hurt less, but refitting downstream layers would be costly, so I'm weighing a cheaper skip-layer test instead.

Which middle layer to drop is still open. The M9 jump after layer 8 looks like it matters, and each test would take one pass over 1024 ids to read the shifted features.

I'm considering a cheap test: drop one middle layer and check whether the features still carry enough signal. If a drop costs no more than about 0.1% on the 1024-id subset, I'd do the full 1496-id read, refit the pool, install it, and verify, which is roughly three or four passes per layer. Layers 3, 5, and 6 are the candidates to screen first. Downstream layers were fitted to the original upstream outputs, though, so I'll need to keep the residual-stream drift in mind after a removal.

I'm weighing whether dropping a middle layer and merging its neighbors could save parameters, but the attention blocks can't be combined that way. Dropping one layer would save about 6% of the parameters, though it could cost a few tenths of a point in accuracy. I'm also not sure how much of the 98.37 is real signal versus noise in the cross-validation estimate.

I'm checking the cross-validation estimate against the test score, since the 20-fold linear result sits close to 98.31. For the kernel model's 98.37 to hold up, I'd want the final CV to stay around 98.2 or higher, though with this sample size a meaningful chance of falling short remains.

Working out whether the base model's test score of 98.31 is a reliable anchor, since the CV estimates carry their own noise. The shared hard sentences between base and this model should make the paired difference more trustworthy than either absolute number.

I'm estimating what the test score should be for this kernel head, using the base linear head's 98.31 as the anchor. The kernel head beats the base head by about 0.17 points in cross-validation, which would put the expected test score near 98.5.

Checking whether the kernel head's paired gap over the base head holds up, I'd expect the test score to land around 98.2 to 98.4, which is a comfortable margin. Dropping a middle layer would trim the parameters but push the risk of falling short up to roughly 5 to 10 percent. Before deciding, I want to screen whether any single layer drop costs less than 0.1 percent on the 1024-id subset, where the baseline sits near 97.88.

I'm narrowing down which middle layers to screen, since removing layer 0 or 1 would break the residual carry. Layers 3 through 6 look like the reasonable candidates, because the per-layer change in the residual stream is roughly uniform and nothing singles them out.

I'm weighing dropping layer 5 or layer 3 as a cheap screen, since each candidate costs one pass over the 1024 ids. I'm leaning toward testing orig layers 4 and 6, because published work on BERT layer dropping suggests middle layers are less safe to remove than top ones, and layer 8 already looks important here.

I'm weighing whether to drop layer 5 first as a screening test, reading the hidden states at layers 0 and 8 across the first 1024 ids. I'm also second-guessing whether it's worth trying to drop a layer at all versus reducing parameters some other way.

I'm weighing a quick test of dropping layer 5 or layer 7, though layer 7 sits right before the layer-8 jump and could hurt more. I'll screen layers 3 and 6 first, and check the per-layer change in the sentence means, using the existing features for layers 4 through 9.

I'm going to look at what base_feats.npz contains, since per-layer data there would let me rank which middle layers change the residual stream least before picking any to screen.



A smaller lam hurts, and neither k-means anchors nor an extra linear term recovers the loss, so I'm measuring each layer's effect on the sentence-mean representation to pick one middle layer to drop and refit the pool head for.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nb=np.load('base_feats.npz'); print({k:b[k].shape for k in b.files})\n"
}
```


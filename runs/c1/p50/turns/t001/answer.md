## reasoning summary

I'm checking notes.md first in case this is a restart, then weighing the strategy. The model is a 12-layer BERT with carry heads in layers 2, 6, 10 and an aggregation layer 11, at 98.31% test accuracy against a 97.8% requirement, so the margin is thin. I'm now setting up the goal of minimizing k times the parameter count.

I'm working out the parameter cost per layer, roughly 2.37M for attention and the output bias, plus about 1537 per feed-forward unit, and wondering whether the feed-forward size can drop to 1. A single layer would put the total near 26.8M. The open question is whether so few layers can still reach 97.8%, given that mean-pooled states from any single layer only reach about 96%.

I'm weighing two routes: keep the existing layers and prune what contributes least, then refit the ridge head, or build a compact design that keeps only the first few layers and pools into CLS with an averaging head. The catch is that mean-pooled features from a single layer top out near 96% on practice data, and the 1496 practice sentences give only a thin cross-validation margin over the 97.8% bar.

Now I'm working through the cost of narrowing the feed-forward layers: at full width, the FFN takes about two-thirds of each layer's parameters. I'm noting that twelve layers with a near-zero FFN would cost about the same as four full-width layers, roughly 52.8M either way. The concern is that removing FFN units damages the representations, so I'm considering whether a closed-form compensation with ridge on the activations could recover them.

I'm now weighing whether attention heads could supply a linear bypass to stand in for the removed FFN units, since a self-attending head would pass a linear function of the input through. But the heads are already committed, so that route is closed, and I'm turning back to the main question of how few layers can still reach the accuracy bar.

I'm considering which hidden states to use: ridge CV on mean-pooled features from any base layer is a cheap way to check how much accuracy a given depth can carry. The plan is to keep layers 0 through j and let the final kept layer's attention average the earlier states into CLS, so the pooled features are the right ones and not the layer's usual output.

I'm thinking about the simplest truncated design: keep the first few layers, and turn the last kept one into a pooling layer whose uniform attention averages the token states into CLS. The FFN in that layer would be switched off, which leaves its units as wasted parameters since m has to match across layers.

I'm weighing whether a pooling layer built around the averaging head is worth its cost of 2.37M plus 1537 per FFN unit. I'm also considering putting that averaging head into the last kept layer instead, having a repurposed head write the ridge scores into reserved dimensions. That would mean carving 8 dimensions out of a space that the existing model already uses in layers 2 through 10.

I'm now looking at whether the final CLS state can carry the pooled information without an extra layer. The pooler only reads the CLS vector, and since CLS is itself an attention aggregate in the last layer, I'm considering whether making some of its heads uniform could fold the mean of the token states straight into it.

I'm noting that the last layer only needs to produce the CLS output, so it can act as a pooling layer and its attention can be rewritten freely. The effective feature is then a function of the mean of the input to that layer plus the CLS state. With k layers kept, the deepest representation actually used comes from k-1 transformer layers, unless the last layer's FFN adds further processing.

I'm weighing whether a ridge fit on mean-pooled token states can get near the 97.8% bar, since that single-layer feature only reaches about 96%. The combined early, mid and late layers reached 97.8 to 97.9%, and the original CV of 97.8% gave 98.31% on test, so I'd want the CV to land at or above that level.

I'm now listing options for going past a linear ridge on mean-pooled token states, such as concatenating means from several layers, using the CLS state, or getting nonlinearity out of the last layer's FFN, which would only act as a fixed random-feature expansion before the linear pooler.

I'm considering using the pooler layer itself as a nonlinear step: a tanh over a projection of the CLS vector, with the classifier fit by ridge on those features. The projection could come from PCA or random directions, which might add a little accuracy over a purely linear ridge, though with 1496 samples in 768 dimensions the gain is probably small.

Now I'm estimating the parameter cost of different layer counts and FFN widths, from the full 12-layer model at 109.5M down to a single layer at 26.8M. Dropping a layer saves at least 2.37M, and each FFN unit removed saves 1537 per layer kept. The open question is how few layers can still reach 98% in cross-validation, so I'll read the mean-pooled hidden states for all 1496 practice sentences across the base model's layers in one forward pass.

Now I'm planning to check which of the 768 dimensions in each layer's output are the reserved carry channels, since those were fit on the practice sentences and would leak into any cross-validation. Once I've excluded them, I'll see whether the remaining layers still carry enough signal to reach the accuracy bar.

Next I'll read the mean and CLS hidden states for all practice sentences at every layer of the base model, in two forward passes. Then I'll run ridge CV per layer and per combination in numpy. The truncated model only matches the base model's representations if the kept layers keep their full FFNs, so I still need to measure how much narrowing the FFNs costs.

I'm weighing how to score each FFN unit: plain magnitude pruning would likely hurt a lot, but the refit ridge head only needs the pooled features to stay separable across the 8 classes, which is a more forgiving bar. Even a fairly distorted representation might still separate them well, and even a bag-of-words style mean could land somewhere around 93%. For ranking units, I'm considering their contribution to the output, roughly the mean activation magnitude times the norm of their output weight column, but that needs intermediate activations, and the forward pass only exposes hidden states and attention, so I'm working out how to get at the per-token values.

I'm checking whether computing FFN activations from layer inputs in numpy would count as reimplementing the forward pass, since the attention output needed for that isn't available from the hidden states alone. Working out the attention in numpy would clearly be that, so I'm leaning toward treating it as off-limits. Closed-form ridge statistics on the existing activations are fine, though.

Pondering whether the FFN units can be scored without their activations, I see that weight-based scoring is an option, though the attention output needed for exact activations isn't available here. I'm also considering whether setting the FFN width to near zero everywhere would work, or whether the harness would reject a zero-size layer. Keeping one unit per layer would cost only a negligible amount of parameters.

I'm considering whether a bag-of-words style classifier might work without the FFNs: mean-pooling the layer-0 embedding outputs with a single attention layer, then fitting ridge on that 768-dim vector. Words like "means" for definitions or "premium" for payments could separate the 8 insurance classes fairly well, though the accuracy is hard to predict.

Multi-head attention in one pooling layer could be a cheaper route than it first looks: with the query vectors set to zero, every head averages its values uniformly, so the 12 heads together give a full 768-dimensional mean of the input, provided the value and output projections combine to identity.

I'm weighing whether nonuniform attention pooling could do better than a plain mean, since the single-layer mean sits around 96% and I need roughly the 97.8% bar that a multi-layer combination reached. I'll first pull the per-layer data to see whether keeping the early and middle layers intact is enough.

I'm thinking about having each of the early layers write its mean into a reserved channel, so the final pooler sees all the layer means at once. The catch is that each carry layer gives up one of its 12 heads, which slightly disturbs its normal behavior. I'd rather fit the carried scores per layer and then fit the pooler jointly over them.

I'm thinking about the LN scaling issue: each layer's normalization rescales the whole vector by a per-sentence factor, so the reserved channel's contribution from each layer would be distorted differently for each sentence, which could break the joint linear model I'm trying to carry.

Now I'm looking at how the layer-norm scaling affects a carried channel: the reserved dimensions are small and barely move the mean and variance, but each layer's normalization still multiplies them by a per-sentence factor. I want to measure whether combining means from a few early layers actually beats a single layer before I commit to that design.

I'm planning to pull the mean and CLS hidden states for every practice sentence at each layer of the base model, then run ridge CV per layer and per combination. I'm also checking whether the mean pooling includes the CLS and SEP tokens, since uniform attention would average over all of them.

I'm planning the pruned model around masked mean pooling over the non-pad tokens, which should differ only slightly from the plain mean. The layer choice is open: keeping a contiguous early range is the default, though dropping middle layers might also work through the residual stream. After that I'd narrow the FFNs, use an attention pooling layer at the end, and fit a ridge head on the edited model's final CLS state. The open question is that a ridge fit and CV on the same practice data could overstate accuracy.

I'm still looking at the leakage question: ridge CV on fixed features is safe, but if the carry heads were fit on labels, the CV needs more care. To keep things clean, I'd make the final CLS hold an unsupervised linear transform of the means and do all the fitting in the pooler. That caps the pooler at 768 dimensions, so combining three 768-dim layer means would mean first compressing each one with unsupervised PCA down to about 256 dimensions.

I'm checking whether the layer means can be combined by letting the residual stream carry earlier layers forward, which would mean fitting ridge on the concatenation of means from different layers.

I'm going to check the working folder next, including whether notes.md already exists and how the directory is laid out, along with the pass directories. I'll pull the data from the base model afterward.



## message

I'll start by checking the working folder for prior notes and the current state of edits and structure, then plan.

## tool_use: run_python

```json
{
 "code": "import os, json\nprint(os.listdir(WORK))\np = WORK/'notes.md'\nprint(p.read_text() if p.exists() else 'no notes')\nprint(edits()[:5], len(edits()))\nprint(structure()['layers'], structure()['m'])\nprint(pass_dirs())\nprint(param_count())"
}
```


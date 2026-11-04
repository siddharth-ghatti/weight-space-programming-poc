## reasoning summary

I'm weighing whether the compensation terms, around 4.9 in norm, could hurt given that the residual stream norm is only about 15 to 20 in BERT and the Gaussian mean estimate may be off. For units with pre-activation means around -1 to -2, the expected GELU output works out to roughly -0.1 to -0.16.

I'm checking whether the compensation magnitude holds up: summing over the dropped units gives a vector of norm around 5, and the GELU bias is negative for most units, so the direction is plausible even if the size is uncertain. The real question is whether the outlier dimensions in BERT's layer-norm output skew the Gaussian mean estimate for those units.

I'm wondering whether the mean estimate is the real weak point, since BERT's outlier dimensions in the layer-norm output probably have nonzero means that the Gaussian proxy ignores. Estimating that mean properly isn't straightforward, so I'm considering whether to skip compensation altogether and just compare accuracy empirically, though that would make it hard to separate the effect of pruning from the effect of compensation.

I'm checking whether the proxy's mean estimate is off enough to matter: for units with pre-activation means around -1 to -3, the expected GELU output is only about -0.01 to -0.08, so the compensation size could be reasonable. The comparison to the hidden-state norm, roughly 15 to 25 with a few outlier dimensions, suggests a shift of about 4.9 before the output layer-norm is noticeable but not dominant. I'm also weighing a half-strength compensation as a compromise.

I'm now picturing BERT's attention layer-norm as x = γ ⊙ normalized(v) + β, and wondering whether the outlier dimensions come from large γ and β values in specific dims. If so, the Gaussian proxy already captures them through β in the mean and γ in the variance, though any consistent offset in the normalized values could still be missed.

I'm reconsidering whether the mean estimate is the real problem: if the proxy's means are biased, the compensation adds a wrong vector, but dropping it leaves out a term of about the same true size. The proxy at least gets the direction partly right, since GELU outputs are mostly small and negative, so I'm inclined to keep full compensation as the principled closed-form choice.

I'm going with the Exp 1 run now, reading the edited ids p0000 through p1023 and keeping both the mean and CLS hidden outputs across all layers.



The edited model's compensation vectors have norms around 2–5, which fits with most GELU units sitting slightly negative. I'm now running Experiment 1 on the first 1024 sentences, collecting mean and [CLS] activations at every layer.



## tool_use: run_python

```json
{
 "code": "ids=json.load(open(WORK/'ids.json')) if False else None\nimport json\nids=json.load(open(WORK/'ids.json'))\nprint(json.dumps(ids[:1024])[:200])\n"
}
```


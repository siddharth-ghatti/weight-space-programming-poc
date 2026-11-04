## reasoning summary

I'm weighing the cost of the deep option, which lands around 95M parameters and is ruled out. The shallow option with one or two layers at width 1 comes in around 27–29M and sits near 97.1% accuracy, so I'm checking whether that readout can hold up.

I'm looking at whether the 52–56M range with FFN pruning at 256–512 wide could reach the ~98% CV I need, since the shallow 27–29M option sits too close to the 97.8% threshold. I want to see how accuracy trades off against width per layer, and whether dropping some middle layers from the 9-layer set keeps the gain.

I'm weighing whether a middle-depth option can reach the ~98% CV I need, since the shallow 27–29M models sit too close to the 97.8% line. I'm also wondering whether attention-based pooling on shallow layers could pull out more context, though that feels doubtful. Then there's the observation that the test set looks easier than CV, which could just be noise at this sample size, and since falling below threshold is costly, I want to keep a safe margin above 98%.

I'm now weighing a cheaper deep configuration, trimming FFN width and dropping some layers to keep the parameter count down. For the width reduction I'd need to pick which hidden units to keep, but I can't reproduce the attention step in numpy, so I'm considering whether the layer's output could stand in as a proxy for its FFN input.

I'm considering whether I can get the exact FFN input without reimplementing attention, by editing a copy of the model so one layer's FFN contributes nothing and reading the resulting hidden states. That would give a cleaner signal for choosing which FFN units to keep, though it still has to be checked against the cost of the edit.

I'm reconsidering whether I can prune the FFN without the exact attention inputs, so I'm looking at using the residual stream as a stand-in for the FFN input and scoring units by their activation spread on those proxy inputs.

I'm noting that the Kwon et al. approach prunes FFN neurons post-training using Fisher gradients, which I can't use here, though closed-form least-squares tuning could work if I had the activations. Since the readout is ridge or kernel, I'm wondering whether that more aggressive pruning would hold up, but that's uncertain and would take several passes to test. I'll first check whether dropping layers is the cheaper route.

I'm considering an alternating-layer structure, such as keeping layers 0, 2, 4, 6, 8, and 10 plus the pooling layer, to see whether that keeps the deep-model gain at lower cost. Dropping whole layers saves more parameters than trimming FFN width, and the base reads suggest the linear quality is fairly flat across depth.

I'm checking whether the carried M0 scores can be combined with the M9 features in a kernel readout, and how much that adds over M9 alone.

I'm considering whether the pooling layer can be redesigned so the CLS token leans on the mean of the layer-11 states instead of its own, since its residual connection is fixed. One option is to make the attention output dominate the CLS input, so the mean carries most of the final representation.

I'm thinking about whether the pooling layer's CLS residual can be overwhelmed by a large gain on the mean, since removing it outright would mean an attention head that subtracts the CLS value, which seems too complicated. Then I'll settle which experiment to run next, with 22 passes left and the final readout still needing the edited model's CLS states.

I'm weighing which layer-dropping pattern to test first, starting with a set that removes some middle layers from the 9-layer stack. I'm also considering an FFN-pruned deep stack at a width around 768, though the parameter cost of the deeper configurations is what I need to check against the budget.

I'm sizing up the parameter budget for the pruned options, where cost scales with depth and FFN width. The FFN part is the big unknown, so I want to test a width of 768 across all twelve layers to see the full accuracy curve in one pass.

I'm going to bump the test up to m=512 to see whether it holds, and only go lower if the features survive. For the importance scores I'll need per-unit activation statistics, so I'm pulling token-level hidden states from the base model across all 13 layers on 64 sentences. The FFN input isn't directly available, so I'm weighing whether hs[L] or hs[L+1] works better as a proxy, or averaging the statistics from both.

I'm working out the proxy for the FFN input: since attention changes the residual only slightly, the normalized previous hidden state, scaled by this layer's first LayerNorm gain and bias, should approximate the input. Then I'll compute each unit's pre-activation on about 2,000 tokens.

I'm weighing whether to score FFN units by their GELU activation spread times their output weight norm, or to skip GELU and use just the mean and standard deviation of the pre-activations, which is a cleaner Gaussian approximation.

I'm weighing a least-squares refit of the kept units' output weights against the full FFN output, which would be a stronger criterion than picking units by variance alone. Ranking by variance times output-weight norm is the simpler greedy choice, so I'll start there and only add the refit if the ranking looks unreliable.

I'm planning the next stage: a pass to read token-level hidden states across all layers for 64 stratified sentences, then an experiment keeping all 12 layers at width 768 with mean folding, compared against the base model's per-layer kernel results, followed by a layer-dropping test.

I'm weighing whether to test pruning the full 12-layer stack at a fixed width, since one pass would then show the feature quality at every depth. From there I'd decide which depth to keep, and whether to carry the layer-1 features forward alongside the deeper ones.

I'm running a pass to pull token-level hidden states across all layers for 64 stratified sentences, which I'll use as the base for the importance scores.



Shallow layers plateau near 97.2%, so I'll aim for a deeper stack made cheaper by narrowing FFNs, and I'm now reading token-level hidden states to pick which units to keep.



## tool_use: run_python

```json
{
 "code": "import json\nR=np.load(WORK/'base_reads.npz'); y=R['y']; ids=R['ids']\nrng=np.random.RandomState(1)\nsel=[]\nfor c in range(8):\n    ic=np.where(y==c)[0]; sel+=list(rng.choice(ic,8,replace=False))\ntok_ids=[str(ids[i]) for i in sorted(sel)]\njson.dump(tok_ids, open(WORK/'tok_ids.json','w'))\nprint(json.dumps(tok_ids))\n"
}
```


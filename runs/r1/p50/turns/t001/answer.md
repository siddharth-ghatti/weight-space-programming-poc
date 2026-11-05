## reasoning summary

I'm checking notes.md first in case this is a restart, then looking through the working folder. I'm also sorting out which approaches the rules allow: closed-form math on the weights and activations is fine, such as dropping layers and refitting the classifier head with ridge regression or least squares on pooled hidden states, since that's a direct solve rather than iterative training.

I'm weighing the closed-form options: ridge regression and least squares are allowed, LDA too, but logistic regression is out since it needs iterative fitting. I'm thinking about reading each layer's CLS hidden states over the practice paragraphs and checking how well a linear head on them predicts the labels, then fitting the head to match the base model's logits instead of the labels, since the labels are in-sample and would overstate fit.

I'm weighing whether to narrow the FFN by picking units based on activation magnitude times output weight norm, and folding the dropped units' mean contribution into the bias. I'm also considering a least-squares reconstruction from kept units, but getting intermediate activations would mean reimplementing the forward pass, which isn't allowed, so I'll lean on weight-based criteria and measured effects instead.

I'm working out how few layers could be kept while still reaching the 76.5% accuracy target after refitting the head. The floor is about 39.77M parameters, with each kept layer costing roughly 2.37M for attention plus 1537 per FFN unit. Keeping zero layers would leave the <s> representation constant, so at least one attention layer is needed.

I'm considering whether a mean-pooled representation would help, since the head only reads the CLS position, and I'm looking at making the CLS token attend uniformly over the sequence in the last kept layer. Zeroing the key weights and bias would make all attention scores equal, so the CLS output would become the average of the value projections across the non-padded tokens.

I'm weighing whether a uniform-attention head in the last kept layer could turn the CLS output into a mean-pooled summary, since the classifier only reads that position. The catch is that the residual and the FFN after it would still reshape that output, so I'd need to check how much the CLS state stays linear in the mean hidden vector. I'm also thinking about fitting the classifier head in closed form on whatever the last layer's CLS produces.

I'm considering whether a linear probe on each layer's CLS and mean-pooled states, fit in closed form with cross-validation, would show how much task signal sits at each depth. Mean-pooled states at an early layer might already carry much of the signal, while the CLS state there probably carries less. If the probe on a layer's CLS works, that's the same representation the edited model would produce once the later layers are dropped.

Fitting the head to the base model's logits, not the labels, should avoid the in-sample inflation, since the encoder layers already saw these practice examples and their CLS states look better than they would on new data. If the pruned model reproduces the base's outputs closely, it should carry over the base's 77.5% test accuracy with less risk than label-fitting.

I'm weighing whether the pruned model can match the base's test predictions closely enough to keep accuracy above 76.5%. The test split is roughly 400 examples, so one point is only about four predictions, which makes this tight. I'm thinking agreement with the base model's predictions is the metric that matters most, and a linear probe's accuracy alone is hard to judge in advance.

I'm planning to first read the base model's outputs on all 1000 practice examples, collecting logits and CLS and mean-pooled hidden states at every layer. Then I'll run offline probes to see how many layers I need to keep, and test layer-drop options with a refitted head, aiming for agreement with the base model's predictions.

Now I'm weighing FFN pruning by weight-based scores, since the FFN input isn't available to measure activations directly.

I'm weighing whether a data-driven proxy for FFN unit activations, computed from the layer's input hidden states, would count as reimplementing the forward pass. Closed-form math on activations the forward passes return seems allowed, but the boundary is unclear enough that I'm hesitating.

I'm weighing whether computing the FFN's intermediate activations from the layer's input hidden states counts as closed-form math on returned activations, or whether it amounts to rerunning part of the forward pass. The mean and variance of those pre-activations would give a usable ranking, but the boundary is unclear enough that I'm hesitating.

I'm considering whether dropping the FFN output in a layer, so the layer returns just its normalized attention state, would give an exact read of that state. The layer norm wouldn't be cleanly invertible, so I'd still need to work out whether that reading is usable.

I'm thinking about getting exact FFN inputs for a layer by building a probe model: strip that layer's FFN and set its second LayerNorm to identity, so the last layer's token output recovers the attention-normalized state. That state can then be mapped back to the pre-LN input with the first LayerNorm's scale and shift. The catch is that token-level reads are capped at 32 ids per pass, so the GELU statistics would need many passes.

I'm weighing whether computing GELU of the first projection on these token states counts as reimplementing part of the forward pass, since that would be the step that turns the exact per-token inputs into intermediate unit activations.

I'm checking whether a numpy-side read of FFN unit activations counts as closed-form math on returned activations or as reimplementing part of the forward pass, and leaning toward something lighter. I'm estimating the parameter budget for keeping one or two layers at this width to see how much room there is.

I'm considering whether a single kept layer could be reshaped so the CLS position computes a useful bag-of-words summary through its attention heads. The one-layer parameter count looks small enough to leave room for that, though a linear probe on it might fall short of the 76.5% target.

I'm considering whether mean-pooling value projections of the raw embeddings through uniform attention could give a bag-of-words feature that a linear classifier uses directly. It seems plausible to land near 70%, which is probably short of the 76.5% target, so I need to check it empirically. I'm also noting that heads fit on deeper, fine-tuned representations may look better in-sample than they will on new data, while shallower layers were changed less, so their cross-validated estimates should be closer to honest.

I'm going to check whether notes.md exists before anything else, in case this is a restart.



## message

I'll start by checking the working folder for prior notes (in case this is a restarted session) and looking at the data.

## tool_use: run_python

```json
{
 "code": "import os, json\nprint(os.listdir(WORK))\np = WORK/'notes.md'\nprint(p.read_text() if p.exists() else 'NO NOTES')\nprint(structure()['layers'], structure()['m'], edits())\nP = practice()\nimport collections\nprint(len(P), collections.Counter(x['label'] for x in P))\ntoks = tokenize([x['text'] for x in P])\nL = [len(t['ids']) for t in toks]\nprint(np.mean(L), np.percentile(L,[50,90,99,100]))\nprint(P[0])"
}
```


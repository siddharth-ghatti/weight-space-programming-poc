## reasoning summary

Dropping layer 6 costs about 0.16 points on the 1024-subset, so the full run might land near 98.2. That's before any refit of layer 7's FFN. I'm weighing whether to refit layers 7 and 8 to recover some of that, which would mean exposing layer 7 in the new structure and fitting its output to the original layer 7's hidden state, but I haven't settled on the target yet.

I'm considering refitting layers 7 and 8 to make up for dropping layer 6: layer 7 would take layer 5's output as input and be fit to reproduce the original layer 7 hidden states on the 64 exposure tokens, with its attention left as is and only its FFN output adjusted. Layer 8 would then get the same treatment, one exposure pass after another.

I'm weighing whether to refit layer 7's FFN directly, choosing 1024 of its own units by energy and ridge-fitting the output to the drift-corrected base h8 target. The heldout error at layer 7 was about 0.077 before the drop, and I'd guess it rises to around 0.3 without a refit, maybe falling to about 0.15 with one, though I can't tell yet how much that helps classification.

I'm working out the drift-corrected target for the layer 7 refit: the pre-LN value that would reproduce the base output, so the FFN output is fit to that value minus the input. I'm also weighing whether to refit layers 7 and 8 in sequence, which would take about five passes and leave some budget, versus screening a different layer to drop instead, since dropping layer 6 already cost about 0.16 points.

I'm considering whether a refit of layer 7 could be done from its existing exposure data, but that data was captured with layer 6's output as input, not the pruned input I'd have now, so a new exposure pass would be needed. Another option is compensating in layer 5 instead, so its output stands in for the skipped layers 5 and 6 together, which might be cheaper to set up.

I'm considering using layer 5's existing exposure data to fit its FFN toward the base layer 7 hidden state, which would stand in for the skipped layer 6 without a new pass. A single per-token FFN can't reproduce layer 6's attention mixing, so this would only capture the part that's predictable from each token, with ridge shrinking the fit back toward layer 5's original weights.

I'm thinking about fitting layer 5's FFN to the base layer 7 hidden state, so it stands in for skipping layer 6, and I can check the heldout error against the no-refit baseline using the existing exposure data without another pass. Layer 7 was refit on input close to base h7, so this only works if layer 5's output lands near that target, and the per-token error will decide whether it's worth a classification screen.

I'm checking the layer 5 exposure data to see whether its FFN can be fit straight to the base layer 7 hidden state, skipping layer 6. The plan is to compare the refit's heldout error against the no-refit baseline and against layer 6's own error of about 0.080, without running another pass.

I'm going to test a few ways of picking which layer 7 units to keep, weighing energy-based selection against correlation with the skip target, and comparing both against the current keeps.



Dropping layer 6 without refitting costs about 0.16%, so I'm refitting layer 5's FFN on its existing pass_30 exposure to mimic the missing layer's output. Checking the heldout token-level error in numpy first.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport lib5\nfrom lib5 import *\nimport inspect\n# which pass is exposure for which layer?\nfor k in range(25,34):\n    m=json.load(open(f'pass_{k}/meta.json')); print(k, m.get('layers'), m['structure']['layers'], [f for f in os.listdir(f'pass_{k}')][:6]) if False else None\nimport os\nfor k in range(25,34):\n    m=json.load(open(f'pass_{k}/meta.json')); print(k, m.get('layers'), m['structure']['layers'], sorted(os.listdir(f'pass_{k}')))\n"
}
```


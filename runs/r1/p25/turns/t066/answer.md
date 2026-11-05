## reasoning summary

Upstream error at m=1536 is roughly double, and the unpruned layer-5 FFN on edited inputs gives h6 R2 of 0.956, below pass12's 0.979 at m=2048. The corrected target only recovers to 0.9556, matching the unpruned FFN, so correction mostly offsets pruning loss. I'm now working out what the final h6 R2 at m=1536 comes to.

Pruning at m=1536 gives an h6 R² of about 0.955, against 0.979 at m=2048, so the error is roughly 2.1 times larger. Sequential correction at layer 5 recovers the pruning loss but not the upstream error from layers 0 through 2, which seems to dominate. I'm weighing whether more data or correction in layers 1 through 4 could shrink that upstream error enough.

I'm weighing whether dropping a layer and keeping a larger m could beat the m=1536 setup, since dropping layer 3 or truncating to five layers both looked worse on earlier runs. I'm also checking whether a least-squares correction on the next layer could offset dropping layer 4, and estimating the test-set flip rate from the logit perturbation to judge how much risk is acceptable.

I'm estimating how flips depend on margins: test margins are probably lower than practice margins, given the 77.5% test accuracy. To model the flip rate, I'd take the pairwise perturbation distribution at m=1536, which is about 1.5x the m=2048 spread, and compare it against each item's margin.

I'm estimating the test flip rate from the perturbation spread: at m=2048 the expected perturbation is about 0.2, so flips scale with the density of test margins near zero. Calibrating against practice gives roughly 2.2% predicted against 1.5% observed, which is close enough given the signed perturbations. If test items cluster more densely near the boundary, the flip rate at m=2048 could reach around 3%.

I'm estimating the test flip rate at m=2048 with the same margin-density model, which gives about 12 flips against 18 at m=1536. With a roughly 1-item bias toward the base model being right, the chance of losing more than 1% of test accuracy comes out somewhat lower at m=2048 than at m=1536.

I'm estimating the test-set risk from the same crude flip model, which gives roughly 15% for m=1536 and 8% for m=2048 of losing more than 1% of accuracy. I'm not sure of the test size, though; if it's around 400 items the variance is large, and I still need to check that the dataset split I'm recalling is right.

I'm asking what the biggest remaining fidelity gains could be: more data for the least-squares fits in layers 0–2, sequential correction, or leaving the CLS path alone, since it already fits almost exactly. The layer-0 error at m=2048 is the starting point, so I'm checking whether a larger first-layer FFN fit with more tokens could reduce it.

Choosing the 32 longest paragraphs, up to 512 tokens each, could yield roughly 16,000 tokens per pass instead of the current average of about 127 per id. That's around four times the data, with less document diversity but still varied tokens within each paragraph. I'm checking the length distribution of the practice paragraphs first to see whether that's realistic.



At m=1536 the upstream error is about twice that at m=2048, and even an unpruned layer-5 FFN only reaches R² 0.956, so fidelity must come from better fits in layers 0–2. I'm checking the token-length distribution next to see whether choosing the longest paragraphs gives more tokens per probe pass.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nP=practice()\ntoks=tokenize([p['text'] for p in P])\nL=np.array([min(len(t['ids']),512) for t in toks])\nprint('mean',L.mean(),'pcts',np.percentile(L,[50,75,90,95,99]), 'max',L.max())\no=np.argsort(-L)\nprint('top32 lengths sum', L[o[:32]].sum(), L[o[:32]])\nprobe_ids=json.load(open('pass_5/meta.json'))['ids']\nprint('current probe ids total tokens', sum(L[[int(i[1:]) for i in probe_ids]]))\nnp.save('tok_len.npy',L)\n"
}
```


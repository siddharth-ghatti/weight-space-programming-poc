## reasoning summary

I'm treating the exposure as the way to get the attention-residual sum before the LN: I zero the FFN, set the norms to identity, cut the model at layer 7, and read the last token. After that, the rest is just LN and FFN in numpy for that one layer. For the drop-8 run, I'm about to expose L7 using the current upstream setup.

I'm planning the drop-8 run: exposing layer 7 under the current upstream setup, fitting its FFN to the layer-9 target, and then evaluating the reduced structure on 1024 ids. That comes to four calls in total, leaving two spare. The main open question is how much accuracy the skip-refit can recover compared with plain truncation at layer 8, and I don't have the drop-6 numbers to check that against.

I'm weighing the drop-8 experiment against the risk: an estimated 0.3 point loss would leave a thin margin on the test set, so the chance of failing is too high. Still, the two spare passes could cover a cheap test, so I'll decide after seeing the subset results: keep drop-8 only if its subset CV is within about 0.1 of the drop-4,6 result, otherwise revert. That still leaves two passes for installing and verifying the pool.

I'm weighing whether a drop-7 variant might beat drop-8, since it would feed L8 the L5 output instead of h8 and need just one exposure pass. I'll decide between the two once the subset results are in.

I'm weighing whether a skip-refit approach could beat plain truncation at layer 8. Truncating at L8 without refitting gives the pool kernel on h8 directly, and the depth CV for that comes out at 97.91 against 98.37 for the baseline.

I'm checking whether the L5 skip-refit's held-out error was ever recorded, and it doesn't look like it was. Pass_41 could still give a rough estimate for a drop-8 refit of L7, since the upstream setup has changed only slightly since then.

I'm checking the held-out error for a skip-refit of layer 7 toward the layer-9 target, using the existing pass_41 exposure, to compare it against the layer-5 skip result and see whether it's worth the spare passes.

I'm comparing the held-out error of a layer-7 skip-refit aimed at the layer-9 target against one aimed at layer 8, using the existing pass_41 exposure to avoid spending a new one. I also want to see how far apart the h8 and h9 targets are, and to find the exposure pass for the layer-5 skip so I can compare it directly.



Fit_skip uses an exposure pass to fit a layer's FFN against a later hidden state, and pass_41 (an L7 exposure close to the current upstream) lets me estimate offline whether dropping layer 8 is plausible. Next I'll check what passes 37–43 contain.



## tool_use: run_python

```json
{
 "code": "import json,os\nfor k in range(36,45):\n    m=json.load(open(f'pass_{k}/meta.json')) if os.path.exists(f'pass_{k}/meta.json') else {}\n    print(k, m.get('model'), m.get('layers'), m.get('structure',{}).get('layers'), len(m.get('ids',[])), sorted(os.listdir(f'pass_{k}'))[:8])"
}
```


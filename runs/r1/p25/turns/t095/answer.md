## reasoning summary

Further gains from the g2 fitting tweak look exhausted, so the remaining question is which m to pick. I'm weighing an m=1792 candidate and whether to reuse the a5e_1536 input file from the earlier build.

I'm weighing whether the 1792 build is worth it, since mixing in the a5e file from 1536 should still help by widening the input distribution. I could also spend more passes collecting data at L0–L2 for that size. Working through the numbers, my rough risk estimates are about 28% for 1536, 21% for 1792, and 16% for 2048, though those are shaky and depend on how test low-margin density compares to practice.

I'm estimating the flip risk for the test set: at a base test accuracy of 77.5%, low-margin items are roughly a coin toss, so each flip costs about 0.2 expected correct answers. Working through a few flip counts, the chance of an outright loss looks like 36% at 16 flips, 19% at 10, and 10% at 7, though I'm not confident in the test low-margin density.

I'm checking the F1 side next: a few governance flips could move macro-F1 by about 0.006, so the accuracy bar is probably the binding constraint. Given that, I'm asking whether the edits could push test accuracy above base rather than just avoiding a drop.

I'm wondering whether pruning mostly strips out the fine-tuning memorization, which would make flips on practice more harmful than flips on test. I can't tell from here which practice flips come from memorized predictions, so I'm leaning toward m=1792 for now.

I'm settling on a plan for the remaining passes: build and evaluate an m=1792 candidate first, then an improved m=2048 candidate, and pick between them afterward. That leaves six passes spare for further improvements, such as collecting more probe data for the early layers, though that gain looks modest.

I'm comparing the candidate sizes on fidelity and risk, and the choice of m looks like the main remaining lever. I'm about to estimate the L0–L2 errors for an m=2048 build to see how it compares.

I'm going to build the m=1792 candidate and evaluate it, pulling logits and CLS hidden states for all 1000 items. From those I can simulate the flip counts for each candidate size using the measured flip-versus-margin curves, so the choice rests on the relative flip numbers rather than the absolute risk estimates.

I'm now working out the expected flip counts for each candidate size, smoothing the practice numbers with a Gaussian model of the margin change. Before building the m=1792 version, I'm noting that the earlier 1536 build had 22 flips and the 2048 build had 15, though those counts are small and noisy.



Gain-weighted selection didn't help, so the choice of m is the main lever now. I'm saving notes and building an m=1792 candidate with the same recipe.



## tool_use: run_python

```json
{
 "code": "\nwith open('notes.md','a') as f: f.write('''\n## Run 3 (handoff 2, turn ~80)\n- pass17 RESULT C1536: agree .978, low .89, logitR2 .9946, acc .908, F1 .8975; 22 flips (19 right->wrong). m2048(pass12): agree .985, 15 flips.\n- head affine correction (fold into classifier.dense, CV): +.003 agree only -> not worth.\n- postLN-weighted / g2 selection: no gain. Token sigma uniform. Fitting method exhausted.\n- Risk model: test ~400 items, can lose only 4 net. Flips happen at base margin<0.55. Crude fail-risk: m1536 ~25-35%, m2048 ~15-20%.\n- evalh.py: stats(k) helper (needs sys.path.insert(0,WORK)).\n- Next: build m=1792 candidate (build_candidate, ae a5e_1536.npy) -> pass18.\n''')\nimport sys; sys.path.insert(0,str(WORK))\nimport lsbuild\nkeep=lsbuild.build_candidate(1792, base, save_edit, drop_edit, edits, ae_file='a5e_1536.npy')\nimport json; json.dump({str(k):v for k,v in keep.items()}, open('keep_C1792.json','w'))\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep={l:keep[l] for l in range(6)}))\nprint(param_count(), [len(keep[l]) for l in range(6)])\n"
}
```


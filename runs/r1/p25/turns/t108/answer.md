## reasoning summary

I'm checking the sign of the change in margin for low-margin items the base model got wrong, which came out positive, and wondering whether regression to the mean could explain why the edited model's margin for the wrong answer would grow.

The margin shifts for the two groups point in opposite directions: low-margin items the base model got right lose margin, while the ones it got wrong gain it. That pattern fits pruning pulling predictions away from the fine-tuned labels, so the pruned model appears to lose some of the memorized push toward the training targets.

I'm working through the expected test flips for each option, reasoning that without the memorized push on test items the flips should be closer to symmetric noise, with a mean around -0.15 per flip. I'm also weighing whether the per-class shift of about ±0.03 is large enough to justify a correction, and deciding it isn't. The decision analysis still needs the margin density near zero, which I'd estimate from the low-margin items on the practice set.

I'm estimating the test flip rate per option from the low-margin density and the noise level, and the rough numbers come out to about 12.6 flips for the 2048 option, 13.8 for 1792, and 15 for 1536. The gap between options is small under this crude model, so the failure probabilities look like roughly 17–20%, 20–23%, and about 25%. I'm now asking whether anything could substantially shrink the noise at the lower-margin end.

I'm checking how a lower-noise setting would shift the flip count, and estimating that a larger model width at 2560 could cut the margin noise roughly in half, down to about 6 flips. I'm still weighing whether that gain justifies the higher cost against the expected score from the 1792 option.

I'm working through the expected score under each option: about 11.4 for 1792, 9.4 for 2048, 12.9 for 1536, and only 5.4 for 2560. On this crude model the lower-pruning options look worse for expected value, but that depends on how heavily failure is penalized. Since the failure probability rises slowly as the width shrinks, I'm leaning toward more aggressive pruning, though the model is rough.

I'm weighing whether tightening the fidelity at m=1536 could offset the higher flip rate. The cheapest lever is adding one more batch of L0–L2 token data, which might trim the error by roughly 5–10%, but that's a modest gain. The L3 and L4 fits, which rely on only about 4,000 tokens, already have small errors, so they're less likely to be the bottleneck.

I'm now considering whether the CLS path changes things: only the final token matters there, but its attention reads every token's layer-4 output, so errors in upstream token representations still carry through. I'm also weighing a constraint I may be able to exploit: with the same unit count required at every kept layer, I'd want to see whether tying units across layers could reduce error at m=1536.

I'm weighing whether the L0 feed-forward output could be approximated from token identity alone, since its input is largely the word embedding plus position, which might let me cut its error at m=1536.

I'm considering whether the feed-forward output at layer 0 could be shifted onto the later layers' feed-forward blocks, so the early layers would need fewer units. The idea is to free up capacity where the errors are largest, since the deeper layers already look close to lossless even at smaller widths.

I'm settling on the 1792 option as a middle ground: the estimated agreement is about .981 with roughly 19 practice flips, most of the reds coming from lost memorization.

I'm now checking whether a drop of 5 or more test items would fail the bar, since 4 items sits exactly on the 76.5% threshold and could be a rounding edge case.

I'm wondering whether the test set might not be 400 items, but I'm fairly sure it is, based on the split I remember. I'm also checking whether there's some other way to make the flips less likely where they matter, and deciding that the self-ensembling idea doesn't really fit here.

I'm comparing the options on the F1 side as well: a few net governance errors would be needed to break the macro-F1 bar, so accuracy is the constraint that binds first. I'm weighing m=1792 with base-only L5 at roughly .981 agreement against m=2048 with new data at about .986 and 1536 as the alternatives.

I'm weighing whether to spend three more passes on extra low-layer token data before the final builds. A rough power-law estimate suggests that doubling the data would only trim local errors by about 6–7%, which seems too small to justify the passes.

I'm settling on whether to spend the remaining passes on more data or on evaluating final candidates, and leaning toward evaluation since extra low-layer data would only trim errors by a few percent. I'll build the honest candidates and check them, starting with 1536 which still needs a real evaluation pass.

I'm weighing the expected-value math: 1536 comes out ahead on a crude model, since a lower failure risk for 1792 doesn't offset its smaller size advantage, though that depends on how the score treats failures.

I'm weighing whether a reliability-first priority should outweigh the expected score, and I'm checking the margin noise more carefully by reweighting the practice items to resemble the test set's margin distribution, since the base model's accuracy drops from 92.5% on practice to 77.5% on test.

I'm planning a probe pass on the 1536 upstream to get an honest read on the base-only L5 fit, since the earlier 1536 margin-noise figure is still missing. A second probe on the 2048 option with new data would follow, and then one final verification pass on whichever candidate I choose.

I'm setting up the 1536 probe now: rebuilding the first five layers and then overwriting the final layer with probe edits, since its kept units don't matter once the feed-forward is zeroed.



Pruning moves the labels away on both base-right and base-wrong items, which looks like memorisation loss; the per-class bias shifts are too small (±0.03) to fix. I'm probing the L5 inputs of the m=1536 upstream, rebuilt with the new data, to score its base-only L5 fit exactly.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport lsbuild, json\nkeep=lsbuild.build_candidate(1536, base, save_edit, drop_edit, edits, ae_file=None)  # L5 base-only\njson.dump({str(k):v for k,v in keep.items()}, open('keep_C1536b.json','w'))\n# save the base-only L5 edits for later, then overwrite L5 with probe\npre='roberta.encoder.layer.5.'\nnp.save('l5_baseonly_1536_W.npy', W(pre+'output.dense.weight')); np.save('l5_baseonly_1536_b.npy', W(pre+'output.dense.bias'))\nsave_edit(pre+'attention.output.LayerNorm.weight',np.ones(768,np.float32))\nsave_edit(pre+'attention.output.LayerNorm.bias',np.zeros(768,np.float32))\nsave_edit(pre+'output.dense.weight',np.zeros((768,3072),np.float32))\nsave_edit(pre+'output.dense.bias',np.zeros(768,np.float32))\nsave_edit(pre+'output.LayerNorm.weight',np.ones(768,np.float32))\nsave_edit(pre+'output.LayerNorm.bias',np.zeros(768,np.float32))\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep={l:keep[l] for l in range(6)}))\n# check L0-4 edits equal C1536 used in pass17 (L0-4 identical construction)\nprint(sorted(edits()))\nwith open('notes.md','a') as f: f.write('''- pass19 RESULT: a5e_1792.npy saved. Offline exact (reproduces pass18 logits 1e-5). HONEST 1792: full-FFN L5 ceiling agree .983; base-only L5 .981 (19 flips, dm std .29, low .43). pass18 .985 was in-sample.\n- Practice flips are memorization loss (low&right margin drops, low&wrong margin rises) -> test flips more random.\n- pass20 planned: L5 probe on C1536 upstream (L0-4 new data) -> a5e_1536b.npy; keep_C1536b.json; l5_baseonly_1536_W/b.npy saved.\n''')\n"
}
```


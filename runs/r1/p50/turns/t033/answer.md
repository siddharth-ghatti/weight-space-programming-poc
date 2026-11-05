## reasoning summary

The CLS token carries a large constant component in one dimension through layers 1 to 5, so its varying part is small compared with the constant. Layer 5 then does most of the work of pulling information into the CLS token, raising its varying norm from about 2.5 to about 15.

So the final layer's attention from the CLS token is what gathers information across all tokens, which means errors in the token states from layer 4 are what matter for classification. My per-layer pruning error of about 0.2 was measured relative to the full residual, including the large constant dimensions, which may understate the real error. I'm going to recompute it relative to the variation across tokens instead.

I'm now measuring the pruning error against the variation across tokens rather than the full residual, since the large constant dimensions could be hiding the real error. Layer 5's FFN on the CLS token also looks like a major contributor, so I'm weighing whether to refit the layers sequentially against each layer's base outputs, though that is limited by the amount of data.

I'm weighing how to pick the final m, and whether the agreement and mean|dm| numbers at m=1536 are good enough. In-sample agreement of about 97.7% might not meet a 99% bar, so I'm trying to figure out a test-risk criterion I can actually estimate.

I'm treating the practice examples the base got wrong, or that sit near the decision boundary, as a stand-in for the test distribution, with margins scaled down to reflect the lower test accuracy. For each example, a flip would happen when its noise pushes the margin below zero. I want to estimate flip rates under a few scale factors for the m=1536 setup, since the cost of each flip is what ultimately decides the risk.

I'm estimating how many test flips the m=1536 setup would produce, and what that does to accuracy. Each flip costs roughly 0.2 net on average, so the variance grows with the flip count. At 8 flips out of 400 test examples, the chance of losing more than 1% is around 17%, and at 4 flips it drops to about 4%. So the target is to keep test flips to roughly 4-8 or fewer.

I'm estimating test flip risk at the margin: for low-margin flips between the top two classes, base and new predictions are about equally likely to be right, so the net loss per flip is small. The variance is what matters, since each flip shifts accuracy by one example either way. At around ten flips in 400 test examples, the standard deviation of the accuracy change gets large relative to the 1% threshold.

The test flip count needs to stay under about 8-10 out of 400 examples. Around 20 flips would put the accuracy drop near the 1% threshold, and macro-F1 looks slightly more sensitive because of the small governance class.

I'm weighing whether m=1536 is safe enough: its roughly 2.3% in-sample disagreement could become 5-7% on test, which is too many flips. Getting the noise down roughly three times would mean going to a larger m, probably around 2560, since the pruning error shrinks more slowly than the gap I'd need to close.

Next I'm weighing a weighted least-squares fit, where errors count according to how much they change layer 5's attention from the CLS token, rather than treating every hidden direction equally.

I'm noting that a least-squares fit with a metric weighting Q gives the same solution, since each output column is fit independently, so reweighting only helps if it guides which units to keep. I'm scoring each unit by its variance times its effect on layer 5's values in that metric, and selecting by that score.

I'm weighing whether to pick units in layer 5's FFN by how much each one moves the logits, rather than by raw variance. Since the head only reads the CLS token through a small subspace that shifts with each example, a unit's score would be its variance times its sensitivity through that head Jacobian.

I'm considering whether to score units by logit sensitivity instead of raw variance, since only a few directions of the CLS state reach the logits. Computing that sensitivity exactly would amount to a backward pass, which I want to avoid, so I'm looking at a cheaper regression-based proxy.

I'm working out how the logits relate to the CLS state: a ridge readout gives a 768×4 direction matrix, so a small change in the CLS token shows up as a logit error through that matrix. I'd need the layer-normalized change in the CLS token, which I can approximate linearly from the pre-norm residual change and its per-example scale. Still, all of layers 0 to 4 would need to share the same m, and layer 5 alone doesn't account for the whole problem.

I'm looking at whether pruning layers 0 to 4 contributes most of the margin noise, since that run alone gave about 0.44, well above the m=1536 all-layers case. Because layer 5's CLS attention mixes in values from every token, errors in tokens that receive little attention from the CLS token should matter less, so the real question is which parts of the layer 4 outputs that attention actually reads.

I'm weighing whether a smaller pruning dimension could still be workable, since layer 5 is the key layer and the earlier layers carry little of the CLS information. Dropping the middle layers already failed, and attention-only reduction isn't an option here, so I'm narrowing the search to six layers with m somewhere between 1536 and 2560. I want to estimate the margin noise at m=2048 using better compensation, which probably depends on how many tokens I use to fit it, and I'm checking whether the dropped units' contribution can be predicted at all.

I'm going to check how much the held-out error drops at m=2048 with plain least squares. I also want a sequential fit that trains each layer's compensation on the edited model's inputs, aimed at the base model's residual targets, so it can partly correct upstream error. The cost of that formulation is the part I still need to pin down.

I'm considering whether a sequential fit could help: each layer's compensation trained on the edited model's inputs, aimed at the base model's residual targets. Layer 0 needs no probe since its input is exact, and the remaining layers would take five passes. For layer 5, I could fit only the CLS residual, or even target the logits directly, which might reduce the error that's left upstream.

I'm checking whether layer 4's token errors matter for the layer 5 CLS readout, but I don't have per-token numbers for that yet. A simpler lever is more probe data: longer paragraphs would give several times more tokens per pass, which should tighten the least-squares compensation, though likely only by about a quarter.

I'm checking whether the noise criterion is too strict: at m=1536 the in-sample flip rate is about 2.3%, and on test that could plausibly reach 5-7%, which costs too much accuracy. So I need a larger m, and the open question is how much the margin noise drops at m=2048 with the current least-squares compensation.

I'm considering a larger m around 2560 and better compensation in parallel. One data-free option is merging near-duplicate FFN units with nearly parallel input weights, though trained FFNs rarely have such duplicates.

I'm going to measure the noise at m=2048 with the current least-squares compensation first, to get a baseline. For layer 5 I'll fit only on the even-indexed examples and evaluate on the odd ones, so the estimate isn't optimistic.



The `<s>` vector's varying part is tiny until layer 5, where it gathers the paragraph's information through attention over all tokens, so pruning errors in layers 0-4 matter. I'm now calibrating the noise-versus-m curve at m=2048 across all six layers, fitting layer-5 compensation on even-indexed paragraphs only.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'builder.py').read())\nfor e in edits(): drop_edit(e)\ndef build_layer_slice(Lr, m, lam_rel, fit_mask=None):\n    keep = build_layer_zero(Lr, m, lam_rel, fit_mask)  # zeroed dropped cols; slicing removes them anyway\n    return keep\nfk = {}\nfor Lr in range(5): fk[Lr] = build_layer_zero(Lr, 2048, 1.0)\neven = np.arange(1000)%2==0\nfk[5] = build_layer_zero(5, 2048, 0.1, fit_mask=even)\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep=fk))\nwith open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- pass10: 6L m1536 (var-rank + ridge LS comp, lam_rel 1 for L0-4, .1 for L5): agr .977, acc .904, margin noise mean|dm| .256. Flips all in low-margin.\n- pass11: zeroing-sim L0-4 m1024, L5 full: agr .947, mean|dm| .437. CLS has huge constant dim (~22) in L1-5; varying part tiny until layer 5 (varying norm 2.5->15.3) => layer 5 attention over L4 tokens is where info gathers; token errors in L0-4 matter.\n- Risk model: test N=400, need test flips <~2% => need very low noise. builder.py: build_layer_zero(Lr,m,lam_rel,fit_mask).\n\"\"\")"
}
```


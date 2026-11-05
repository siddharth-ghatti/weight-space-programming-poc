## reasoning summary

I'm noting that the greedy pass 29 gave agreement of .977, slightly below the .981 from v1, so the greedy change looks like noise and I'll keep v3 as the saved edits. With 21 passes left and the model around 68.1M params, I'm weighing what to try next, since the ~.98 agreement may be partly in-sample because the compensation was fit on some of the same practice activations.

I'm weighing whether to go smaller: a 6-layer model at width 1024 comes to roughly 63.4M params, about the same as 5 layers at width 1536, while 5 layers at 2048 is slightly bigger at 67.3M. My earlier notes leaned toward keeping width 1536 since the linear score looked flat across 1024 to 1536, and I still have a large budget left to decide whether to keep improving or submit.

Flipped items on test are low-margin, and base is probably right only about half the time there, while the new prediction is right maybe 30% of the time. That makes each flip a small net loss, roughly a quarter of an item on average. Ten flips would cost around 2.5 items, about 0.6%, against a 1% threshold, so the risk turns on variance rather than the mean. I still need to check how the earlier risk simulation handled that.

I'm weighing whether a bigger model would be safer, since m2048 at 6 layers would be about 72.9M params, roughly 4.7M more than the current m1536 setup. Beyond model size, I'm considering whether the compensation could get less noisy with more practice data, given that the learning curve was flattening between 12.6k and 16.9k tokens.

I'm thinking about whether the classifier really needs the full token-level compensation, since only the <s> position feeds the final output. In the last layer, only the CLS token's output matters, so the FFN work on other tokens there is irrelevant, though the model width is global so that doesn't free up much on its own.

I'm now considering whether the L5 attention can be exploited: only the CLS query matters there, so the errors that count are the ones that move CLS values through the value projection. Dropping layer 5 outright looks weak, since the layer-4 CLS probe only agreed about 91.5% of the time. So I'm leaning toward weighting the compensation fit by how much each token's error feeds into the final CLS.

I'm working out whether a full-matrix metric on the layer-4 output error would beat the LN-gamma and sigma weighting I already used, since only the CLS-attended value path actually reaches the final output. A common regressor makes the least-squares solution independent of the metric, so the weighting has to come from the downstream sensitivity itself, not from the fit.

I'm checking whether the proxy gains actually carry through: the layer-4 selection looked about 20% better on holdout, yet the full-model agreement barely moved, so the metric improvements may just be noise at this sample size.

I'm considering whether the flip rate on test can be pushed lower by estimating each item's margin, since the flip probability depends on how close the base logits are, and about 2% of practice items fall in that range. Refitting the head on pooled logits looked risky with only about 1000 samples, so I'm inclined to keep m1536 and put the remaining passes into collecting more probe data on the edited model.

I'm wondering whether the extra probe passes are worth it, since the learning curve was flattening and only gave about a 3% error reduction. I'm also reconsidering whether m1024 or a 5-layer model could work, given my earlier notes said the expected score was about flat between them. My earlier self picked m1536, but with roughly 10–16% failure risk at that size, I'm asking whether the noise can be pushed down or whether I should accept a smaller model.

I'm checking the risk against the test set size: if it's 400 items, the base model's 310 correct answers leave room to lose only 4 net, so the margin for error is tight.

I'm adding that macro-F1 has to stay above 0.6896, a drop of about 0.02 from 0.7096, and since the governance class is small, a couple of flips there could move it noticeably. Each flip on low-margin test items is probably a small net loss, and with around ten flips the expected loss is about 1.5 items with a fair amount of variance, so the risk is mostly about how the flips fall.

I'm estimating that with m2048 the practice flips drop only modestly, to around 16 from about 20, which leaves the test failure risk near 7–9%. The bigger model barely cuts the noise, and I'm wondering whether the practice flip count is itself inflated by in-sample fitting.

I'm wondering whether the practice flip count is itself deflated, since the fitted items and base's training memorization make practice margins look larger than test margins. That would mean test flips could run higher, maybe 8 to 16 instead of the earlier estimate. I'm now looking for a less noisy way to reduce the error.

I'm weighing an end-to-end correction instead, fitting a linear map from the edited model's final CLS hidden state to the base's, since the layer-5 CLS is the only thing the classifier reads.

I'm wondering whether a per-token correction would hold up better, since the errors in short sequences get more weight in the CLS average, and the training ids are all at least 120 tokens long while the test paragraphs average closer to 100.

I'm weighing whether a shorter sequence length changes the error picture, since positional and CLS-token effects could matter more there, though earlier notes suggested CLS errors are tiny. Given how much effort has already gone into noise reduction, I'm leaning toward choosing between m1536 and a smaller alternative like a 5-layer model at m2048.

I'm working through the parameter arithmetic for a five-layer model at two widths, which comes to about 65M and 75M. Dropping layer 4 earlier gave agreement near .933, far below the ~.98 I'm getting from compensation, so layer removal looks poor unless it's compensated better. I'm wondering whether a different layer would be less damaging to remove.

I'm considering whether to fold a linear correction for a dropped middle layer into the next layer's Q, K, and V weights, since its residual input could be replaced by a mapped version of the earlier hidden state.

I'm weighing whether to keep the current v3 edits or revert to v1, since their agreement scores (.977 vs .981) and error sizes differ only by noise. A layer removal with compensation doesn't look promising given the earlier drop-layer results, so the choice is really between these two saved versions and how to spend the remaining passes on a better estimate.

I'm weighing whether to deal with the layer-0 and layer-1 noise, since they account for about half of it, and I'm wondering whether token identity alone can explain the layer-0 feed-forward input.

I'm wondering whether the layer-0 feed-forward input could be approximated from token identity alone, since that would let me compensate for the biggest noise source without new forward passes. But building those inputs from the embeddings would mean reimplementing the model, which I can't do, so the only route left is fitting on activations I've already collected.

I'm weighing whether more probe data would help the layer-0 and layer-1 noise, but the gain looks like only about 2.5% in total variance, so I don't think it's worth the passes. The compensation seems close to its limit, which leaves the choice of model width as the main lever, and the earlier analysis suggested that choice was roughly flat between m1024 and m1536.

I'm weighing whether going up to m2048 for safety is worth it, since it would drop failure risk to roughly 6–9% but cost about 4 points of size reduction. Putting rough numbers on it, m1536 still comes out ahead on expected value, though a smaller width like m1024 looks better on a linear calculation and I need to check that more carefully.

I'm checking whether a 5-layer model at m1536 would be worth it, and it probably isn't, since its size is about the same as the 6-layer m1024 option. The m1024 width now looks like about .965 agreement, roughly 35 practice flips, and a failure chance near 25–35%, which makes the expected score come out about even with m1536. I'm not sure there's a way to make the head more reliable on test without a validation set, so I'll stay with the current plan.

I'm weighing whether the base model's heavy overfitting, with practice accuracy at .925 against test at .775, means that pruning noise mostly regresses it toward the mean. If so, edits that flip low-margin items in mostly random directions might leave test accuracy roughly unchanged. The one thing I'd question is whether my earlier flip-rate estimates are too pessimistic, since in a four-class setting the runner-up is often the correct class when the base is wrong.

I'm wondering whether the base model's calibration on unseen data would tell me if the flips are symmetric. Practice is all training data, so the only test-level signal is the overall 77.5% accuracy, which doesn't settle it.

I'm weighing whether to spend some of the remaining passes on a more realistic check of the noise, since it was measured on practice tokens and compensation was partly fit on those same activations, so the test-side picture could differ. The base model's test accuracy of 77.5% against 92.5% on practice suggests fewer low-margin items on test, which would change the flip count. I'm inclined to try one more improvement idea at moderate cost before settling on a submission.

I'm checking whether the logit errors between the edited and base models have a systematic per-class offset, since a fixed bias correction could cut flips without adding many parameters. That check is cheap to run offline on the saved passes.

I'm going to check whether the edited model's logit errors have a systematic per-class offset, or whether a small linear correction on the logits could reduce flips. I'll test that with cross-validation so it doesn't just overfit the practice set.



pass_29 (greedy selection) scored about the same as v1 (0.977 vs 0.981), so I'm running a cheap offline cross-validation check on the saved logits for a systematic error that a few head parameters could fix.



## tool_use: run_python

```json
{
 "code": "bl=np.load('base_logits.npy'); y=np.load('y.npy')\nfolds=np.load('folds.npy') if os.path.exists('folds.npy') else None\nprint(folds.shape if folds is not None else None, np.unique(folds) if folds is not None else None)\nfor f in ['lg_pass25.npy','lg_pass29.npy']:\n    lg=np.load(f); d=lg-bl\n    print(f,'mean dlogit',d.mean(0).round(3),'std',d.std(0).round(3))\n    # centered dlogit vs centered logits\n    dc=d-d.mean(1,keepdims=True); lc=lg-lg.mean(1,keepdims=True)\n    print(' corr matrix dc~lc', np.round(np.corrcoef(dc.T,lc.T)[:4,4:],2))\n    # CV correction: lg' = lg + lg@A + c  (ridge)\n    rng=np.random.default_rng(0); fo=rng.integers(0,10,1000)\n    for lam in [0,1,10,100]:\n        pred=np.zeros_like(lg)\n        for k in range(10):\n            tr=fo!=k; X=np.c_[lg[tr],np.ones(tr.sum())]\n            B=np.linalg.solve(X.T@X+lam*np.diag([1,1,1,1,0]),X.T@d[tr])\n            pred[~tr]=lg[~tr]-np.c_[lg[~tr],np.ones((~tr).sum())]@B\n        print(' lam',lam,'agr',(pred.argmax(1)==bl.argmax(1)).mean(),'rms',np.sqrt(((pred-bl)**2).mean()).round(4))\n    pred=np.zeros_like(lg)\n    for k in range(10):\n        tr=fo!=k; pred[~tr]=lg[~tr]-d[tr].mean(0)\n    print(' bias only agr',(pred.argmax(1)==bl.argmax(1)).mean())\n"
}
```


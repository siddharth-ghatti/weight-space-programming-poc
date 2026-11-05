## reasoning summary

I'm comparing candidate C at m=1536 against m=2048: agreement is 0.978 versus 0.985, and it's sitting close to the larger size. I'm also working out parameter counts, landing at about 68.1M for the smaller model and roughly 72.9M for the larger one.

I'm checking the test numbers now: base sits at 77.5% against a 76.5% requirement. Agreement of 0.978 on practice implies about 2.2% of predictions flip, and on test I'd expect the disagreement rate to be higher, since the probe was fit on practice activations. Pruning usually costs a little accuracy, and the practice drop of 1.7 points hints at a similar outcome on test.

I'm weighing whether to shrink the model further, say to m=1280 or 1024, or to keep m=1536 and try to improve fidelity. Fewer layers at similar size don't look better, since dropping a layer was lossy. Given the test-set risk and that failing is catastrophic, I need to be confident the candidate passes before trading size for it.

I'm estimating how many test items could flip, and since 77.5% of 400 is 310 correct against a 306 requirement, I can afford to lose only about four net items. Around 9 to 16 flips would be expected, so whether this candidate passes depends on whether those flips split roughly evenly between gains and losses.

I'm estimating the net effect of flips on test: if about 12 items flip and only about a third of those flips are net losses, that's roughly -3.9 items, which sits right at my margin of four. Even the larger m=2048 model would likely lose two or three items under the same assumptions, so the risk here isn't unique to the smaller candidate.

I'm weighing whether 0.985 agreement on practice is enough margin, given that the practice set is only partly held out from the fit and the test flip rate could run higher. The safer choice would be minimal pruning, but the scoring rewards smaller models, so I'm looking for a balance that still keeps the test accuracy comfortably above the threshold.

I'm now trying to size how much test flipping is likely, since the base model is only about 77.5% accurate there and so has many more low-margin items than practice does. My plan is to measure how the logit margin shifts on practice, then use that spread as a proxy for the risk on test.

I'm now estimating how the logit margins shift on practice, since the base model is only about 77.5% accurate on test and so likely has many more low-margin items there, which would make the flip rate higher than practice suggests. I'm also considering whether an alternative fix at the same size, such as refitting the head against the edited final-layer hidden states, could tighten fidelity.

I'm considering whether a linear correction on the final-layer CLS hidden states could tighten fidelity at the current size, by folding a fitted map into the classifier's dense layer and checking it with cross-validation so it doesn't just overfit the 1000 items.

I'm considering a sequential correction: refitting each layer's output projection on the edited inputs, which I've done for the final layer already, and checking whether the earlier layers can be handled the same way with token-level probe passes. With eight passes left, I'm weighing how much that gains against the size difference between the candidate options.

I'm now looking at whether the early layers can be compressed more gently, since L0 through L2 carry the larger errors at m=1536 while L3 through L5 have plenty of room. I'm considering whether the L3 feed-forward block could partially absorb the residual error left by L2, given that LayerNorm sits between them.

I'm weighing whether the early-layer error can be absorbed by the next layer's feed-forward block, but the earlier sequential refit only gave weak gains, so that route looks unpromising. I'm now considering whether to stick with m=1536 and tighten fidelity through a head refit, then checking how the practice drift would translate to the test set.

I'm breaking down the practice disagreements to see how many are base-right-to-wrong versus the reverse, since the base model is in-sample there and is more often correct. On test the split should be more even, and with 400 items and only about four net items of room, I'm treating the flip balance as the key unknown.

I'm now weighing how much test disagreement to expect, since practice agreement of 0.978 suggests roughly 16 to 20 flips on 400 items, which could easily cost four net items. Even though flips can go either way, for a calibrated model the expected loss per flip is small but positive, so the margin still looks uncomfortably thin.

I'm working out the expected cost of the flips: since they cluster at low margins, the loss per flip is small, so about 16 flips would cost roughly 1.6 items on average, with some spread. That puts the chance of losing more than four items at around 10% for m=1536, though I'm wary that a fine-tuned model may be overconfident on test errors.

I'm estimating that a flip costs little in expectation because the margins are small and the test logits are probably overconfident, but the variance is still large enough that with about 16 flips the chance of losing more than four net items comes out near 21%. For the m=2048 candidate the same reasoning applies, though its lower disagreement rate makes the tail a bit thinner.

I'm checking whether the flip rate on test could be higher than on practice, since the practice items have much larger margins and the base model fits them in-sample. If most practice margins are large, the practice disagreement understates what test will show.

I'm now trying to estimate how many test items sit at low enough margins to flip, since practice margins look much larger because the base model was fit on those items. Practice errors may be a rough guide to the hard items on test. I'm considering a perturbation model where a flip happens when the logit noise exceeds the item's margin, which would let me estimate the flip rate from an assumed test margin distribution.

I'm checking how many practice items sit below a margin of 0.5 and below 1, and what the spread of logit perturbations looks like, so I can estimate how many test items would flip.



Candidate C (m=1536) agrees with the base model 97.8% of the time with 0.908 practice accuracy, and the m=2048 fallback reaches 0.985 agreement and 0.916 accuracy. Next I'll estimate how many test predictions might flip, since test margins are smaller than in the practice data.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nfrom evalh import *\npb=Lb.argmax(1)\nsrt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]\nprint('margin quantiles', np.quantile(marg,[.01,.05,.1,.2,.5]))\nfor k in [12,17]:\n    L=np.load(f'pass_{k}/logits.npy'); pe=L.argmax(1)\n    # perturbation of pairwise logit differences\n    D=(L-L.mean(1,keepdims=True))-(Lb-Lb.mean(1,keepdims=True))\n    pw=np.abs(D[:,:,None]-D[:,None,:]).max((1,2))\n    # perturbation of margin between base top and base second\n    i2=np.argsort(Lb,1)[:,-2]\n    dm=(L[np.arange(1000),pb]-L[np.arange(1000),i2])-marg\n    print(k,'dm std',dm.std(),'mean',dm.mean(),'|dm| q',np.quantile(np.abs(dm),[.5,.9,.99]),'maxpw q',np.quantile(pw,[.5,.9,.99]))\n    flip=pe!=pb\n    print(' flips',flip.sum(),'base right->wrong',(flip&(pb==lab1)).sum(),'wrong->right',(flip&(pe==lab1)).sum(),'wrong->wrong',(flip&(pb!=lab1)&(pe!=lab1)).sum())\n    print(' flip margins',np.round(np.sort(marg[flip]),2))\n    # is |dm| correlated with margin?\n    print(' corr |dm| vs marg',np.corrcoef(np.abs(dm),marg)[0,1], ' |dm| for marg<1', np.abs(dm[marg<1]).mean(), 'marg>=1', np.abs(dm[marg>=1]).mean())\n"
}
```


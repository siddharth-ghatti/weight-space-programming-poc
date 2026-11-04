## reasoning summary

The skip refit to h9 came out at .196, which is worse than just using h8 as a stand-in for h9 at .175, so it's no real improvement over truncation. Truncating at L8 cost .46 in the depth CV, so dropping layer 8 is off the table. Next I'm wondering what the L5 skip (dropping layer 6) error looks like for comparison, and which exposure pass I should use for that.

The remaining lever seems to be m, the hidden width. Dropping it to 896 would save about 2.8% of parameters at a cost of roughly 0.15–0.2%, which doesn't seem worth the risk. The pool layer needs around 1000 anchors to work, so m has to stay near 1024 for it.

I'm wondering whether the pooler's tanh layer could absorb the pooling step, since it already gives a 768-unit nonlinearity on the CLS token before the classifier.

I'm checking whether the pooler's tanh units could take over the kernel features, since they already give 768 nonlinear units on the CLS token before the classifier. The classifier can combine them linearly, and a few units near the linear regime could carry the M0 term through. That would leave the pool layer's attention for mean pooling, but its FFN would no longer need to do the kernel work.

I'm considering folding the pooling into the last content layer instead of keeping a separate pool layer. Only the CLS position matters at the end, since the pooler reads only that token, so layer 8's attention could handle the mean pooling for CLS and free up the extra layer.

I'm weighing whether the pool step's nonlinear features could instead come from a later layer's per-token output, but mean pooling a linear function only gives back the mean of the inputs. Pooling layer 8's tokens would therefore lose its per-token nonlinear processing, which means a layer after 8 is still needed to pool M9.

I'm weighing whether to fold the kernel work into the pooler's 768 tanh units, which would leave the pool layer's FFN free, but m is shared across layers so that only helps if m drops. The content layers seem to need m around 1024 to keep error low, since 768 units gives noticeably worse held-out error than 1024 or 1536.

Now I'm checking whether dropping an early layer like 1 or 3 would be cheap, since skipping layer 4 cost nothing but skipping layer 2 looked costly. The pooler's kernel capacity already matches what the anchors give, so there's no accuracy gain there.

I'm wondering what dropping layer 7 would cost. Layer 8 would then take h7 as its input instead of h8, so the mismatch is similar to the one I already measured for skipping layer 7 into h9 (.196, against .175 for using h8 as the stand-in). My rough guess is that it would cost somewhere around 0.3 to 0.5, so I'd like to check the actual numbers before deciding.

I'm comparing the L5 skip error against the drop-6 numbers, since the cost there looked small even though the skip refit was similar. The depth results show L8 to L9 gives a big jump, while L4 looks surprisingly strong, so the layers clearly don't all matter equally.

I'm weighing whether to stop pruning here. Further drops look likely to cost around 0.3% or more, and my margin is about 0.5%, so the risk isn't worth it. The current model should land near 98.3–98.4 on test, and dropping layer 8 would put it near 97.9–98.0, which makes failure much more likely. I'll stop and finalize this configuration.

I'm settling on the current structure: m=896 would cost roughly 0.15–0.25 for under 3% of parameters, so that's not worth it either. Next I'm going through the final steps, starting with the edited L7 read pass and checking that the linear M0 term carries over exactly from the embedding mean, before running the full CV.

I'm going to run the drop-46 configuration with a couple of regularization values, then fit the pooler on all 1496 examples and install it. After that I'll check the edited model's logits against the numpy predictions on the first 1024 ids, with a few spare read passes left over.

I'm going to check poolfit.py to see how the pooler was installed last time.



Skipping layer 8 would act like truncating at h8 (−0.46%), leaving almost no margin over 97.8, so I'm stopping pruning at [0,1,2,3,5,7,8,11] with m=1024 and finalizing next by reading the remaining ids, running full-1496 CV, then fitting and verifying the pool layer.



## tool_use: run_python

```json
{
 "code": "print(open('poolfit.py').read())"
}
```


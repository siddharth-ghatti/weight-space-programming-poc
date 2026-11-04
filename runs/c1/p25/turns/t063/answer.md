## reasoning summary

I'm checking the "start-only" fix rate for model 8, which comes out to 10 of 33 start errors fixed, about 30%. The start-test error rate is lower than in cross-validation, so I'm working out what that does to the expected accuracy under my simulation.

I'm weighing k=9 against k=10: k=9 carries roughly a 14-19% failure risk plus implementation risk, while k=10 is only about 1-3%. The size gain from k=9 is modest, so I'm inclined toward k=10 as the safer option, though I'm also considering whether attention-weighted pooling could make k=9 safe.

I'm weighing whether attention-weighted pooling could let a single mean-style slot approximate what layer 8 computes per token, and it looks like it can't: the per-token nonlinearity of layer 8 gets lost once everything is aggregated at CLS. I'm now turning to a third idea, using two carry channels to hold more information.

I'm weighing whether extra gains could come from tuning the kernel for model 8, such as adjusting the scale parameter, trying RBF combinations, or whitening the features per dimension. Each tweak might add a small improvement, but with selection bias that gain looks marginal, and I'm also reconsidering the risk of the k=9 option.

I'm weighing whether skipping an early layer like 0 or 1 might cost more than dropping layer 4, but the earlier layer-dropping experiment suggests per-layer cost is roughly uniform. So removing layer 4 from the 9-layer stack might leave something close to the 8-layer model's quality.

I'm weighing whether the test set being easier overall would also make the candidate-only errors rarer than my simulation assumed, since I scaled them by the CV ratio rather than the test-easiness gap.

I'm now weighing whether a merged design could help: folding the pool readout into the last BERT layer. Still, a separate layer 9 can't do per-token work and CLS readout at once, so I'm sticking with k=10 as layers 0 through 8 plus the pool slot.

I'm considering whether the pool slot could be merged into the last kept layer instead. If the final layer's FFN only has to serve the CLS position, then its attention at CLS could read the hidden-8 tokens directly, which would make k=9 workable without a separate pool layer.

I'm asking whether the pool slot's softmax attention could make its readout nonlinear per token, so that it captures something layer 8's FFN would otherwise supply.

I'm considering whether the pool slot's attention could do a soft max-pooling over tokens: if the query-key scores are scaled large, each head would focus on the tokens with the strongest evidence for a given class, which might capture something a plain mean would miss.

I'm considering whether the pool slot's attention could do a max-style pooling over tokens by fitting per-token class scorers from the mean-based readout. Checking token states directly would take about 24 reads, which is too costly, so I'm weighing a cheaper route of two passes followed by a readout fit.

I'm considering whether the pool slot could use multiple heads to do a soft max over tokens, giving max-style class evidence alongside the mean. I can't check that without per-token states, which would take too many reads, so I'd only be guessing whether it helps. Given the 98.31 test result against the 97.8 threshold, I'm leaning toward k=10 with a margin to spare, with CV around 98.05.

I'm weighing whether to spend the remaining budget on a soft max-pooling variant of the k=9 pool slot, with both mean and max-style heads, and I'd only keep k=9 if its CV reaches about 98.0. Otherwise I'd fall back to k=10. Counting the reads, the k=10 build and verification take about 4 passes and the k=9 exploration another 4, so the total of 8 fits within the 17-pass budget.

I'm settling on the uniform-pooling k=10 build first as the safe path, so I can get it running end to end. The soft-max pooling idea for k=10 could add more margin later.

Still writing the pool slot forward pass...

Still writing the pool slot's FFN weights...

Still working out the pool slot's FFN and output weights: the reserved dimensions carry the dual coefficients, and the pooler rows decode them as centered differences.

Still working through the pool slot's output: the argmax comes down to the carried linear part plus the kernel-style FFN contribution per class, so the joint ridge has to fit both together.

Still writing the fitting setup: the kernel features from the layer-8 reads and the two carried-channel feature blocks, with the centering handled inside the carry weights.

Setting up the joint fit now: the kernel features from the layer-8 reads plus the two carried-channel blocks, solved as one dual ridge system.

Still working through the joint kernel: the carried channels add a per-sample scaled kernel term on top of the FFN kernel, so the dual form still gives a valid combined kernel.

Still working out how the pool slot's output scores combine: the carried residual and the FFN contribution both feed into the final scores before LN2, so the argmax is unchanged by the shared normalization.

Working out how the pool slot's output scores come together: the argmax is taken over the final score per class, which combines a kernel-weighted term from the layer-8 reads with the carried linear scores, each scaled by the sample's own factor.

Working out the magnitudes now: the carried scores are tiny, so the reserved-dimension values after LN1 come out around 2 times the raw score, and the FFN term has to be scaled to match that.

Still working out the carry scale: I'm setting the head weights to a small multiple of the fitted coefficients, so the carried values stay near their original magnitude while the fit runs in normalized score units.

Scaling the FFN contribution by the same ε as the carried heads keeps the decoded score at ε·f(x), so the pooler gain G can be set to 1/ε. That keeps the tanh inputs in a range where it doesn't saturate, since saturated outputs would tie at 1.0 and let the first index win the argmax.

Checking the float32 accumulation risk on the dual coefficients next, since the sum over 1496 terms could lose precision if the alphas grow large.

Checking whether the pool slot's attention covers all unmasked tokens uniformly, which means the padding mask has to be handled correctly. I'm also wondering whether the mean-hidden read includes the CLS and SEP tokens, since that affects what the mean is averaging over.

Next I'm checking the carried-score measurement: the reserved dimensions at hidden layer 9 hold the layer-2 and layer-6 scores, and the new pool layer's output should line up with those scores scaled by some factor κ2 and κ6. I'll regress the centered pool-layer reads on the centered start-model scores to pin down those two factors.

Checking whether the per-token scaling still leaves the carried scores proportional to the same centered vectors, I'm reasoning that the layer-6 contribution is identical across tokens, so only the per-token ratios differ before the pool averages them.

Next I'm checking whether anything leaks into the other computations: the FFN weights skip the reserved dimensions, and the LayerNorm statistics include them but they are tiny, so that looks fine. I also expect the earlier single-coefficient fit residual of 0.111 to drop to about zero once the two carried scores each get their own scalar.

Checking whether the carry head at layer 6 reads the reserved dimensions, since if its value weights touch those columns then s6 would pick up a function of s2 and break the two-coefficient fit. I'm planning to zero those columns explicitly when writing the new weights, and to keep the value bias at zero as in the start model.

Checking the bias term now: the carry value bias needs to be -D2·μ2 so the centered score is produced exactly, otherwise a constant offset shifts the scores. The intercept itself is uniform across classes, so it doesn't change the argmax.

Next I'm checking the kernel features on the u-space reads: z-scoring with the mean and standard deviation from all 1496 reads, with the FFN implementing the same GELU kernel so the two stay consistent. I also want to compare cross-validation on u against the M-space version before settling the structure. Pass A will use the layers [0..8, 11] with width 3072.

Next I'm checking which reads I still need from the edited pool-slot model: the last-layer CLS state for the 1024 ids first, then the other 472, since the layer-2, layer-6 and layer-9 means already sit in the saved base reads.

I'm planning the readout for the last two layers, then the logit checks: the first pass reads the layer-9 and layer-10 states, and the rest follow after that.

I'm weighing the choice of attention scale α. Setting α to 8 keeps the pooled value mostly the scaled mean, with the CLS state contributing only a small share, though the mean's norm is smaller than the CLS vector's, so the balance needs checking.

Setting the value projection to α times the identity, with output as the identity, so each head simply averages the tokens. Query and key weights go to zero, so attention weights stay uniform.



The k=10 build comes first, with the pooling slot at layers 0–8 plus original layer 11, and I'll test whether a richer k=9 pooling can close the gap if budget allows. Step 1 is two read passes to get the kernel inputs and the per-sentence carry scales κ2 and κ6.



## tool_use: run_python

```json
{
 "code": "for e in edits(): drop_edit(e)\nalpha=8.0; P='bert.encoder.layer.11.'\nZ768=np.zeros((768,768),np.float32); z768=np.zeros(768,np.float32)\nsave_edit(P+'attention.self.query.weight',Z768); save_edit(P+'attention.self.query.bias',z768)\nsave_edit(P+'attention.self.key.weight',Z768); save_edit(P+'attention.self.key.bias',z768)\nsave_edit(P+'attention.self.value.weight',(alpha*np.eye(768)).astype(np.float32)); save_edit(P+'attention.self.value.bias',z768)\nsave_edit(P+'attention.output.dense.weight',np.eye(768,dtype=np.float32)); save_edit(P+'attention.output.dense.bias',z768)\nsave_edit(P+'attention.output.LayerNorm.weight',np.ones(768,np.float32)); save_edit(P+'attention.output.LayerNorm.bias',z768)\nsave_edit(P+'output.dense.weight',np.zeros((768,3072),np.float32)); save_edit(P+'output.dense.bias',z768)\nsave_edit(P+'output.LayerNorm.weight',np.ones(768,np.float32)); save_edit(P+'output.LayerNorm.bias',z768)\nprint(set_structure(layers=[0,1,2,3,4,5,6,7,8,11], ffn_keep=3072))\nprint(edits())\nopen('notes.md','a').write(\"\"\"- Exp F result: m=2560 M9 kernel 97.36 vs base M9 97.71-97.81, base M8 97.6 (1024 subset). FFN pruning dead -> m=3072.\n- Risk sim (paired vs start test 98.31): k=9 (M8) P(fail)~14-19%; k=10 (M9) ~1-3%. DECISION: build k=10 [0..8,11] first.\n- Pool slot L11 design: q=k=0 (uniform), V=8*I, O=I, LN1 g=1 b=0, FFN off, LN2 g=1 b=0 -> final CLS = u = LN(x_cls+8*mean). Read u for all 1496 (passes 9,10).\n\"\"\")"
}
```


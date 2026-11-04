## reasoning summary

I'm checking whether there's a cleaner way to frame the economics, since the parameter count splits into a fixed embedding term plus a per-layer term. The attention block per layer comes to roughly 2.37M parameters, while the feed-forward block at width 3072 is about 4.72M.

I'm working out the parameter counts for the deep and narrow variants, where a 10-layer model at width 3072 lands near 95M and a 9-layer one near 88M. A 12-layer model at width 2048 would be about 90.6M, which doesn't fit the budget. I'm also weighing whether multi-layer readouts could recover some of the accuracy lost by narrowing the FFN at width 1536.

I'm weighing whether least-squares refitting of the FFN output weights from probe activations would be worth it. Ridge regression on about 2,000 tokens might fit, but I doubt it would generalize to other tokens. The bigger problem is that pruning layer L changes the inputs to every later layer, so sequential probing would need roughly ten passes, which is too expensive.

I'm comparing a few checkpoint depths against the parameter budget. The 9-layer option at width 3072 sits near 88M and the 10-layer one near 95M, and I'm weighing whether 7 or 8 layers with multi-layer readouts could match them.

I'm checking whether dropping an early layer would keep the 9-layer model's accuracy near the 10-layer one, since the gap between them is only about 0.4% and could just be noise. I'm considering a few layer-drop patterns to test in a single pass, starting with a candidate that skips layer 4.

I'm considering whether dropping layer 4 could keep the 9-layer model's accuracy close to the 10-layer one, and I'd test that with a single pass. Layer 11 is already a pooling layer in the start model, so I'm also looking at whether it could serve as the pool slot.

I'm working through how the readout would be built: the start model's carry heads sit in layers 2, 6, and 10, and their ridge values were fit to the start model's joint readout. Each one reads the mean of its layer's input and writes scores into reserved dimensions, so the final model's carries would need to be refit to whatever layers survive.

I'm checking how the carry heads from layers 2, 6, and 10 would be refit for the surviving layers, since layer 10 won't be kept. The reserved dimensions have to be excluded from the reads in layers 2 through 10, and their LayerNorm gains set so they pass through untouched.

Now I'm working out how the pool slot's FFN would read the CLS position's attention output, which means weighing how to build the kernel readout from that slot's hidden state.

I'm now looking at how the readout would work if the pool slot's FFN handled the kernel, starting from whether the CLS hidden state at the final layer can recover the slot's attention output with the FFN turned off.

I'm setting the pool slot's first LayerNorm to identity-affine so its FFN input is just the normalized CLS attention output, which lets me read that exact state back from the final CLS with the FFN switched off. From there I'd fit a Nyström kernel readout on that state, with the units and output weights set so the residual feeds the class scores through the second LayerNorm.

I'm checking how the reconstructed pool slot's FFN output would combine with the attention residual before LayerNorm, and whether the reserved dimensions would stay clean enough for the pooler to read class scores.

I'm checking how the carried scores from the reserved dimensions would combine with the attention output at the pool slot, since the final CLS state is just a normalized sum of the two. The pooler's linear read would then need to separate the kernel scores from whatever content the attention output already holds in those same dimensions.

I'm settling on how the carried scores and kernel scores share the same reserved dimensions, so the final values are just their sum. The pooler then reads each class from its dimension after removing the common offset, and since every class shifts by the same amount, the argmax still picks the right one.

I'm checking whether the reserved dimensions stay clean enough for the pooler to read class scores, since the attention output at the CLS position might already hold content there. The kernel scores and carried scores would share those dimensions and simply add, and a common offset or scale across classes wouldn't change which class wins.

I'm settling on a Nyström-free version: with m=3072 I can use all 1496 practice sentences as kernel centers, making the readout exact kernel ridge. I'm checking whether the quadratic kernel built from normalized features still works with the LayerNorm-normalized attention state.

I'm checking whether the kernel can be computed on the unnormalized, z-scored state, since dividing by its norm isn't something the FFN can do. Using the raw dot product with a quadratic term would sidestep that, because the LayerNorm already fixes the norm of the hidden state.

I'm checking whether the kernel can be evaluated on the z-scored features without the L2 normalization, since the FFN can't divide by a per-sample norm. Then I'm considering swapping the quadratic kernel for a GELU feature map applied directly to the projected features, and fitting a ridge readout on those, with cross-validation to see whether it holds up.

I'm checking whether the GELU features on the centers can be fit with ridge to get out-of-fold accuracy, which would count as a legitimate readout design check. It's the same kind of closed-form fit as the earlier numpy prediction check. I'm also wondering how the pool slot would behave once its readout is built this way.

Next I'm weighing whether a kernel readout is worth the extra complexity compared with a linear one. The numbers suggest it is: the kernel version lands around 98.1%, against roughly 97.5% for the linear prefix and 97.8% for the start-model approach. Then I need to decide which layers to keep, and I'm considering dropping layer 4 as a first test, though I'd want to read all 1496 examples to get a reliable estimate.

Now I'm weighing whether to start with a 1024-example check against the base models on the same subset, or to build the candidate model right away and read its activations across all 1496 examples in two passes. The second option would also let me evaluate the actual readout, so I'm sketching how the pool slot's attention would compute the mean.

I'm working out the pool slot's attention so that every head uses zero queries, which makes the attention uniform across all tokens and turns the output into a mean of the hidden states. The value and output projections are set to pass that mean through unchanged, scaled by a large gain, so the attention residual mostly carries the averaged token state into the layer norm.

I'm weighing how the reserved dimensions behave after the uniform attention, since each token's layer norm rescales them differently and that complicates any closed-form view of them. Rather than unpick that, I'm settling on a simpler route: reading the final CLS state and fitting the readout on the full 768-dimensional h_attn, reserved dimensions included.

I'm weighing whether the carried ridge scores would leak into the cross-validation, since they're fit on all 1496 sentences. A cleaner option is to carry a fixed projection that doesn't depend on the labels, such as the top principal components of the M1 features, so the readout can be checked fairly.

I'm weighing a carry projection that doesn't depend on the labels, such as the top principal components of the M1 features, so the CV stays honest. Fitting the ridge scores on all sentences would leak label information into every fold and make the estimate look better than it is. I'm also wondering whether more than 8 reserved dimensions would close the gap to the linear M1 readout.

Checking whether the carried projection can be refit inside each CV fold, so the readout's evaluation doesn't leak label information. The carry is linear in the layer-2 input states, which I already have from the base reads, so refitting per fold looks feasible.

I'm making the carried values honest for cross-validation: the carry is linear in M2, so I can fit it per fold alongside the kernel block. In the model itself, the carried term and the kernel's contribution would both be written into the same reserved dimensions and summed.

I'm checking how the carried term and the kernel contribution will sit together in the same reserved dimensions, since per-sentence LayerNorm scale factors could shift their relative weight. The sentence-dependent scaling is small, roughly a few percent, so the combined score should stay close to what the fold-wise fit predicts.

I'm weighing whether to pool from hs[9] alone or add a carried term for the linear M1 signal. Dropping the carry would lose about 0.35 points, which seems worth keeping. I'm also thinking about whether k=9 versus k=10 is the right trade, and whether dropping layer 4 is worth testing, before I commit to the build.

I'm planning to compare a few candidate layer sets, reading the mean hidden states for each from the base reads so I can estimate cross-validated accuracy before building the pool slot. The first option uses layers 0 through 8, the second drops to layers 0 through 7, and the third skips layer 4 while keeping the rest.

I'm checking a few candidate layer sets, dropping layer 4 or another layer, by comparing cross-validated accuracy for each. I plan to score them on the full set using the linear layer-2 readout and try a few kernel scalings and regularization strengths, including plain z-score normalization, so I can see what realistic accuracy looks like.



Layer-to-layer changes look similar across the middle of the network, so that doesn't single out the free layers. I'm now running fold-wise CV for depth 8 and 9 on the full set, with the carried projection refit in each fold.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\ndef zsc(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, lam=1e-2, seeds=(0,1,2)):\n    out=[]\n    for s in seeds:\n        f=folds(y,5,s); S=np.zeros((len(y),8))\n        for i in range(5):\n            tr=f!=i; te=f==i\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n        out.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(out)),4)\nd=len(keep)\nfor j in [8,9]:\n    Zj=zsc(M(j)); Cj=Zj@Zj.T/d   # no L2 norm\n    Zn=Zj/np.linalg.norm(Zj,axis=1,keepdims=True); Cn=Zn@Zn.T\n    for lk in [1,2]:\n        Zl=zsc(M(lk))/np.sqrt(d); Gl=Zl@Zl.T\n        for c in [0.5,1,2]:\n            print(j, 'lin M%d'%lk, c, 'L2norm', kcv(Cn+0.5*Cn**2+c*Gl), 'zonly', kcv(Cj+0.5*Cj**2+c*Gl), kcv(Cj+0.5*Cj**2+c*Gl, lam=3e-2))\n"
}
```


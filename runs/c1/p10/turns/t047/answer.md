## reasoning summary

Even after layer 0 prunes 512 units, layer 1 still shows a 14% centered change, which is large for dropping the least important sixth of units. Either the importance estimate is poor or the mean compensation is off, since the shift of 0.114 should be near zero if compensation worked. Now I'm checking what the mean hidden activation of layer 1 looks like.

I'm weighing whether computing the FFN activations from the token hidden states would count as reimplementing the forward pass, which the rules forbid even in numpy. It looks like a partial forward computation, so I'm leaning toward treating it as against the spirit of the constraint and sticking with the earlier proxies despite their crudeness.

I'm wary of computing the FFN pre-activations from the token hidden states, since that reproduces part of the model's forward pass in numpy, which the rules rule out. So I'll stick with the earlier statistical proxies, though I want to check which moments they actually rely on, since the layer 1 shift suggests they may be off.

I'm weighing whether to pull token-level hidden states across all layers, since they'd give enough samples to estimate a covariance for the FFN input. The catch is that the FFN input is the post-attention LayerNorm output rather than the layer's hidden state, so those reads wouldn't be an exact match for what the proxy needs.

I'm weighing whether to avoid reproducing the model's forward pass by computing FFN activations from hidden states, and I'm settling on a conservative plan. Accepting the drift from pruning the FFN may be the safer route, since the subset leave-one-out drop is sizable, and I'll check whether final accuracy holds up on the full data. Dropping FFN pruning entirely at k=10, m=3072 would still give about 13% size reduction with less risk.

Now I'm weighing a way to fix the bias compensation with the reads I already have: the base and edited mean hidden states at each layer for the same sentences. Since layer 0's input is identical in both models, the change at layer 1 should come only from the pruned FFN units, so I can use that difference to correct the compensation.

I'm weighing whether to compute the FFN input from the read hidden states, but the attention step would mean reimplementing part of the forward pass, which the rules forbid. A cleaner option is to use the base and edited mean hidden states at each layer, since layer 0's input is identical in both models, so the layer 1 shift reflects only the pruned FFN units.

I'm weighing a head-to-head between leaving the FFN alone at m=3072 for a safe ~13% size cut, or pruning to m=2816 with a better proxy. Each read pass counts against my budget of five, and the final head fit plus a verification predict pass need at least two of them.

I'm weighing whether to keep the m=3072 setup, which needs no more pruning, or push to m=2560 by reading the remaining 472 sentences on the edited model. Going with m=2560 means one more read, then the head fit and a verification pass.

Now I'm working out how the final CLS comes out of the pooling layer: it's an attention step over the layer-8 outputs with uniform weights, followed by a layer norm, and I need to check how the reserved dimensions are handled in that normalization.

Now I'm weighing the reads needed for each option: m=3072 costs two reads for the final CLS plus a verification pass, which leaves two spare. Pruning further to m=2560 would also cost size-wise, though I'd first want the remaining sentences read to settle the head.

I'm estimating the full-data score for the m=2560 model: the subset drop of about 0.5 points from base 97.75 to 97.27 suggests roughly 98.2 on the full set, and with head realization losses around 98.0. The test-set noise is the main risk, so I'm weighing the probability that the test score falls below the threshold against the size savings.

I'm weighing whether to add the embedding mean M0 to the final CLS, since it's unaffected by pruning and might lift the head's score by about 0.3 points. The layer-8 pooling only sees layer-8 outputs, so M0 would have to ride in the reserved carry dimensions, which gives me just 8 slots. I'm considering sending a linear ridge score of the layer-1 mean through those slots.

Wondering whether the 8 reserved carry dimensions could hold ridge scores from a kernel model on the embedding mean M0, since the pooler can only take the 768 final CLS dims and all of M9 already fills them.

I'm noticing the pooling layer's FFN is currently zero, but with the m units already paid for, it could act as a second, nonlinear stage over the CLS vector. That would let me fit a small two-layer head with GELU features on top of the final representation.

I'm considering whether the FFN, now that it's paid for, can act as a kernel machine with one center per training sentence. That would give exact kernel ridge regression rather than the 768-center approximation, which I estimate recovers about 0.3 to 0.5 points. The catch is that GELU features aren't an exponential kernel, so I'd need to check how close they get.

I'm checking whether a GELU feature can stand in for the exponential kernel I want. For large negative inputs it decays like a Gaussian, which is close to an RBF shape, so the approximation may be workable even though GELU isn't exp.

I'm considering whether the GELU features at the pooling layer's FFN can serve as a kernel machine: with m units and one center per training sentence, that could give an exact kernel ridge fit instead of the 768-center approximation. GELU isn't an exponential kernel, but I'm thinking of solving a ridge regression directly on the GELU features, which should still work as a general random-feature head.

I'm checking how the 8 reserved dimensions pass through the pooling layer's second norm: their values stay small, so the 760 other dimensions dominate the mean and variance, and the scores should come out nearly unchanged. The pooler then reads those 8 dimensions with a monotone tanh, so the predicted class should be preserved.

I'm now considering using the FFN's GELU features at the pooling layer as a second kernel stage, with the pooler's reserved dimensions carrying its scores through a near-monotonic tanh so the predicted class is preserved. The remaining pooled dimensions could hold exponential-kernel features at 760 centers, with the classifier reading both.

I'm considering whether the combined head can work in two stages: first fit the FFN's GELU features to the eight class scores, then feed those scores through the reserved dimensions alongside the 760 pooled features. The danger is that scores fit on the training set are nearly perfect in-sample, so a second-stage classifier trained on them would overfit.

Checking whether GELU features at the training sentences as centers can get close to full kernel ridge regression. I can test that offline in numpy on the base activations, since fitting a closed-form head to them is fine. The pooling layer's input is x1, which the edited model would hand me, and since the FFN is zero in that setup, the final CLS is just x1 renormalized.

I'm checking how the LN2 normalization treats the reserved dimensions, which are zeroed by LN1, and whether the 760 remaining values make the mean and variance close enough to a plain identity. If so, the residual difference should be small enough to account for by inverting the final CLS read back to x1.

I'm working out how to recover the pooling layer's input from the final CLS read, since the reserved dimensions are zeroed by LN1 and the normalization stats depend on the whole vector. The reserved dims give me the mean-to-std ratio, but the rest isn't cleanly invertible. So I'm considering changing the pooling layer's LN1 so those dimensions stay live rather than being zeroed.

I'm weighing whether the pooler's input can be made robust to small affine shifts, since the FFN features are computed on x1 and LN2 re-normalizes the final CLS read. The idea is to keep the GELU kernel features tolerant of that mismatch rather than trying to invert LN2 exactly, and to check it with a predict pass.

I'm weighing two plans now: keep the full 3072 FFN units with no pruning, which costs about 13% of the parameters, or prune to 2560 units, which saves about 20% but loses roughly half a point. I'm also wondering whether the FFN head could compensate for what pruning removes.

I'm weighing whether to keep the full 3072 FFN units or prune to 2560, since pruning saves parameters but costs about half a point. To decide, I'm planning offline numpy checks: exact RBF KRR on the full 1496 training set, the 768-center pooler approximation, and a GELU-feature ridge using all training points as centers. The 2560-unit case only has features for 1024 samples, so I'd still need more data to compare it fairly.

I'm weighing whether the pooling layer's attention can carry the M0 signal, but it only sees layer-8 outputs, so that route is closed. The reserved dimensions are limited to 8 values, which caps what an earlier layer could pass forward, and I'd need to check whether the attention heads can still move anything useful into them.

I'm wondering whether the M0 signal can reach the pooler's reserved dimensions at all. The 8 reserved dimensions could carry linear M0 scores, which in the earlier runs added about 0.3 points on top of the kernel head, while linear scores from M2 and M6 added nothing. That gap is odd, since M0 is basically the bag-of-words embedding mean. Carrying those scores forward would need a carry head in layer 0 or 1 that attends over the layer-0 input.

I'm checking whether a carry head at layer 0 could pass M0's signal into the reserved dimensions, but the channel is zeroed by LN at the end of layer 1 and only the layer 2 head can read across it. Since linear ridge on the layer-1 mean already scores about as well as layer 0, I'm wondering whether reading M2 would get most of the same information.

I'm comparing the linear M0 carry result against the M2 and M6 carries, since the M0 gain looks odd given that the other linear carries added nothing. The existing layer-2 and layer-6 carry heads already pass ridge scores for M2 and M6 forward into the reserved dimensions, and layers 2 through 10 keep those dimensions isolated.

I'm checking whether the reserved dimensions in layer 1 could keep their normal BERT values instead of being zeroed by LayerNorm, which would let an M0 carry come from there. The catch is that those dimensions would then also have to stay isolated from layer 2 onward, so I'm weighing whether that change is worth it.

I'm checking whether layer 1 could carry an M0-based ridge score into the reserved dimensions, which would mean a uniform-attention head there writing into them before the attention and output LayerNorms. The catch is that those LayerNorms would also have to leave the reserved dimensions undisturbed and not shift the stats of the other dimensions.

I'm laying out the two options again: keep all 3072 FFN units for safety, or prune to 2560 for a bigger parameter cut at a small accuracy cost. Before deciding, I want offline numpy checks of GELU-feature kernel heads against exact RBF KRR, since that tells me whether the pooler can make up the loss.

I'm sketching how the pooling layer's FFN would write the eight reserved-dimension scores, which then get normalized by LN2 into the final CLS read. The idea is to decode them with a mean-subtracted tanh on those dimensions, so the pooler's output depends only on the differences between the scores.

I'm seeing that the pooler's decoding reduces to an argmax over the eight reserved-dimension scores, since the per-sample scale σ is positive and drops out. That means the FFN output can be written as a GELU-feature kernel ridge with m features, which is exactly realizable in the pooling layer. I'm now checking where x1 enters this, since the FFN reads it directly.

I'm working out how to get x1 exactly: setting LN1's gamma to 1 and beta to 0 on all 768 dimensions makes x1 zero-mean with unit variance, so LN2 passes it through unchanged and the final CLS read gives me x1 directly. The open question is what happens to the reserved dimensions under that setting.

I'm checking whether the eight scores can avoid the reserved dimensions entirely, since the carried editor scores already live there and would mix with mine. Putting the outputs in eight other dimensions would avoid that contamination, but I'd need to confirm those dimensions aren't used elsewhere.

I'm working out how the reserved dimensions behave under LN1 with gamma zeroed there: they come out as zero in x1, so the final read on them collapses to just the mean and scale of x1 over the 768 dimensions. That means the reserved-dimension output is fixed at -m/s, which carries no per-sample information on its own.

I'm working out the reserved-dimension bounds: since those 8 dimensions get zeroed by LN1, the final read on them is just -m/s, and I still need estimates for m and s from the 760 non-reserved dimensions.

I'm pinning down the reserved-dimension output from the normalization: if those dimensions are zero in x1, then m equals minus s times y_res, which gives one consistent equation. The mean of x1 squared is near 1 minus the reserved contribution, so s is close to its constant value unless the reserved values are large.

I'm noting that if the reserved dimensions have zero weight in the first-layer normalization and the centers are zero-mean over the rest, the FFN input reduces to a scaled copy of the final read, with the offset term dropping out. So the pooler can use that directly and I can check the prediction difference numerically.

I'm counting the predict passes I need for the 1496 practice ids, and it looks like a read pass with logits can cover up to 1024 ids, so I'll only need two passes for the full set.

I'm planning to read all 1496 final CLS vectors in two passes, then fit the head and verify it on a read of the edited model. The verification will only catch implementation errors, since in-sample logits will look near-perfect. The open question is whether a stronger GELU-feature head fit on all the data makes the pruned 2560-unit version safe.

I'm now looking at a kernel-feature version: GELU applied to a scaled cosine similarity with a negative bias, which tends toward a ReLU-like hinge as the gain grows. I'm also considering putting all 1496 practice points in as centers and using the remaining units for multi-scale variants.

I'm testing a GELU-feature ridge head with 5-fold CV over the gain and bias, using z-scored features scaled by sqrt(d), to see whether the pruned 2560-unit version holds up.

I'm checking whether the z-scoring can be folded into the first-layer weights, which it can since it's just a per-dimension scale and shift. The RBF form doesn't carry over exactly because the z-scored vectors aren't unit-norm, so I'll test the GELU features directly with cross-validation over the gain and bias.

I'm now writing the z-scored centers into the kernel: a GELU of the scaled dot product with the centers, shifted by the bias, and I'm going to run that check.



The drift grows with depth, so the Gaussian-proxy pruning is lossy. I'll test offline whether a GELU-feature ridge at every practice-sentence center matches RBF KRR on base L9 features, using 5-fold CV.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nfrom lib import folds\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']\nF=lnr(5*d['mean_hidden_L9']+d['cls_hidden_L9'])\ndef cv_gelu(F, y, g, b, lam, seeds=(0,1,2), kind='gelu'):\n    accs=[]\n    for s in seeds:\n        P=np.zeros((len(y),8))\n        for te in folds(len(y),5,s,y):\n            tr=np.setdiff1d(np.arange(len(y)),te)\n            Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n            if kind=='gelu':\n                Htr=gelu(g*Ztr@Ztr.T/dd+b); Hte=gelu(g*Zte@Ztr.T/dd+b)\n            else:\n                Htr=rbf(Ztr,Ztr,g); Hte=rbf(Zte,Ztr,g)\n            T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n            if kind=='gelu':\n                mu=Htr.mean(0); Hc=Htr-mu\n                Wt=np.linalg.solve(Hc.T@Hc+lam*np.eye(Hc.shape[1]), Hc.T@T)\n                P[te]=(Hte-mu)@Wt\n            else:\n                al=np.linalg.solve(Htr+lam*np.eye(len(tr)),T); P[te]=Hte@al\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nprint('rbf', cv_gelu(F,y,0.4,0.003,kind='rbf'))\nfor g,b in [(2,0),(4,-1),(8,-2),(8,-4),(16,-8)]:\n    for lam in [1e-2,1e-1,1]:\n        print(g,b,lam, round(cv_gelu(F,y,g,b,lam,seeds=(0,)),4))\n"
}
```


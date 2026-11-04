## reasoning summary

The shallow model seems to top out around 97.1–97.2% when it uses mean-pooled features, so I need better features to get past that. I'm considering a two-layer setup where the pooling layer's 12-head attention does weighted pooling, and whether both layers could instead be pooling-type with different functions, starting with an attention layer that builds context features.

I'm weighing whether a two-layer pooling design could help, with one layer building per-token local context features and the second pooling them. Since BERT's layer 0 already contextualizes well, I'm asking what richer pooling, like max or squared-mean over hs[1], could capture beyond plain mean pooling. The 64-sentence data is too small to evaluate classification, so any new pooling would have to be built into the model and read from the pool layer's CLS output, which lets me test up to 12 pooling heads in a single pass over the 1024 ids.

I'm working out how the 12 attention heads of a pooling layer could each produce a 64-dim pooled feature, so one pass yields 12 pooling variants to compare. The catch is that 64 dims per head is a tight budget, since PCA-64 alone already dropped accuracy to 92.5%.

I'm weighing a 12-head pooling layer with 64 dims per head, so each head is a different attention pattern over the token states. The candidate patterns are uniform pooling as a baseline, class-evidence weighting based on ridge directions, and position-based pooling that separates the first and second halves of a sentence. Since 64 dims per head is tight, I'm also considering combining heads across PCA slices to cover 256 dims.

I'm wondering whether the attention-only layer 0 result, 95.65% against 95.85% with the FFN, means the FFN is worth keeping for hidden-state quality. The bigger question is whether per-token nonlinear features, the kind a layer's FFN computes, could be averaged as a bag-of-features instead of relying on the pool layer's linear values alone.

I'm weighing a layer A that does per-token nonlinear feature expansion, with an FFN computing φ(x_t) = GELU(W x_t + b), plus attention over neighbors for local bigram context. The pool layer would then average those features, and the open question is whether a linear readout on mean φ can beat a kernel on mean x.

I'm weighing whether a φ built from sharp, word-specific GELU features would turn the mean over tokens into roughly normalized bag-of-words counts, so a linear readout could match a strong text classifier over the 30k vocabulary. A linear model on the projected mean already exists, so the open question is whether the nonlinear expansion adds anything beyond it.

Next I'm considering whether bigram features could come from the attention pattern, since a head that attends to the previous token would fold x_{t-1} into each token's state. BERT's layer 0 already seems to do some of that. The idea is a cheap route, but with only 64 sentences I can't validate it, and I'm still wondering whether a 97.2% CV ceiling is acceptable as the baseline.

I'm weighing the cost of stopping at the 97.2% CV baseline: it sits below the 97.8 CV that the mid-depth options reach, and the actual test score could easily land under that too. Keeping more layers looks necessary. I'm also wondering whether skipping the middle layers, going from early layers straight to 9 and 10, could keep the model cheaper without losing too much.

I'm considering whether dropping alternate or contiguous middle layers could keep a six-layer stack close to the deep-model quality, since middle layers in BERT often change the residual stream only slightly. Reading the mean-pooled features at each depth would show how much signal survives the skips.

I'm considering dropping some of the middle layers and keeping a seven-layer stack, say layers 0, 1, 2, 4, 6, 8 and 10, then reading the mean-pooled features at each depth. The aim is to see whether skipping layers keeps enough signal, since the extra FFN width mattered a lot for the deep layers.

I'm considering whether dropping layers 3, 5, 7 and 9 would hurt, since each remaining layer would then receive an input distribution it never saw during training. I'm also thinking about least-squares refitting of the FFN output weights on the kept units, which the literature suggests could recover much of the loss from pruning.

I'm checking whether a probe built by zeroing a layer's FFN output could expose the FFN input of the original model. The LN2 output would be a known function of the attention output plus the output bias, which might let me recover the pre-FFN states and refit the pruned weights against true activations rather than a proxy.

I'm checking whether the LN2 output can be inverted to recover the pre-FFN attention state. Since the layer norm's mean and scale are known, each token's attention output could be solved up to a shift and scale, with the mean and variance constraints fixing both unknowns, but only one layer per probe and only over a limited set of tokens.

I'm dropping the idea of probing FFN inputs layer by layer, since each layer would need its own pass and only one could be read per probe. Revisiting the goal instead: the pooled feature needs something like the M9 or M10–12 quality, and adding the early layers M0/M1 seems to push it to 98.0–98.2. I'm still working out whether the layer-drop patterns can keep that.

I'm weighing a layer-dropping experiment with a few patterns over layers 0 through 8, since every full layer costs about 7.09M parameters. Shallow configurations at m=1 and m=3072 sit well below the deep six-layer option in size, so I'm leaning toward spending effort on shallow accuracy.

I'm wondering whether the shallow features carry the same information as the deep ones, so I want to compare the M0 and M1+M9 kernel errors to see which sentences depth actually fixes, likely the compositional ones. A simple ensemble of hidden-state means at different early layers didn't help before, so I'm not expecting much from that route.

I'm thinking about IDF-style weighting, downweighting frequent function words and punctuation, and whether an attention score u·x_t could approximate SIF weights a/(a+p(w)). Fitting u by ridge regression on token-level hidden states from hs[0] looks feasible, though it would need many token embeddings.

I'm weighing whether SIF-style weighting, which downweights frequent tokens, could beat the uniform mean, since the ridge readout already reweights dimensions but not tokens. Fitting the attention direction on roughly 2000 token states is feasible but risks overfitting, and I'm not sure the gain would clear 0.3–0.5%.

I'm working out whether the attention-weighted pooling could use the sentence-level ridge direction as a token scorer, since the average of token scores equals the sentence score. That would let a head with u = β·w_c do soft-max pooling over tokens and feed its class-score values into the readout.

I'm considering whether a max-pooling head could work, scoring each token with the sentence-level class direction and taking the strongest evidence. The scale of those token scores can be read off the 64-sentence token data I already have, so I'd set the sharpness without another model pass.

I'm settling on a minimal design: a single pooling layer over the embeddings, with some heads averaging uniformly and others focusing on class-evidence tokens. That would be followed by a pooler and ridge classifier, which should be cheap but still needs about 0.7% more accuracy to reach the target.

I'm picturing the evaluation pipeline for k=1 designs: one pass through the edited model per design, reading the final CLS vector for 1024 ids, then kernel or linear CV in numpy. I'm also considering a single k=1 probe model that reads many pooling variants at once, with each of the 12 heads given its own 64-dim block of the 768-dim CLS output.

I'm settling on the value projection for each attention head: the evidence-pooling heads would carry the eight class-score directions from the ridge fit alongside top principal components of the token states, while the uniform heads would each take a different block of the principal components so that together they cover the top 256 of the means.

I'm considering whether the token-level class scores from the ridge readout could drive the attention pooling directly, and I want to check whether the max-scoring token predicts the label better than the mean. To avoid leakage, I'll refit the ridge on the 1432 sentences outside the 64 held-out ones, then compare mean, max, and softmax-weighted scores on those 64.

I'm weighing the budget now, with about 20 passes left and plenty of code runs available. Layers 0 through 3 at attention-only width might be worth testing, though the earlier results show M4 slipping below the base model.

I'm questioning whether the kernel-on-means ceiling holds for deep features too, since the deep kernel reached 97.6 alone. Dropping FFNs in only the last few layers isn't possible, because the kept layers must share one width m, so every layer pays the same FFN cost.

The deep path looks costly: at m=768 the FFN already hurts, so going deep means roughly 60M or more. The shallow options come in around 27–39M but top out near 97.2% CV. That leaves a choice between a risky shallow design and a safer deep one near 88–95M, and the gap is large unless shallow improvements close it.

I'm checking whether the test accuracy of 98.31% lines up with a test set of 590 sentences, where it would be 580 correct. Working the 97.8% threshold against that size gives a cutoff of about 578 correct, which means at most 12 errors.

I'm checking the threshold arithmetic: at n=295 or n=590, the start model's test errors leave only a point or two of slack, so the bar is essentially the start model's own quality. A shallow design at around 97.2% CV would likely fall short, so I need to reach roughly 97.8% or higher in CV.

Pruning the width to around m=2048 would cut the per-layer cost, so a deep design near 9 or 10 layers could land well under 95M while keeping the kernel-level accuracy I'm after.

I'm thinking about whether the last BERT layer could double as the pooling step, so I'd keep layers 0 through 8 and let layer 8's attention take the place of a separate pool layer. That would give a 9-layer design around 88M, and the earlier numbers suggest it could land near 97.9% accuracy.

I'm wondering whether the last BERT layer's attention could handle the pooling itself, which would let me use the pool slot's FFN units for a Nyström-style kernel readout over the pooled features.

I'm considering a readout that combines a kernel on the final-layer features with a linear term from the first layer, which would mean adding the two Gram matrices together. I want to test a table of kernel-plus-linear pairings across different layer depths, since the depth cost is the real constraint here.

I'm wondering whether pruning the FFN width at depth could recover some accuracy, since m=768 cost about 3% on the deeper features. Each reduction in width saves a fair amount per layer, so I'm weighing whether a moderate cut keeps the accuracy close enough to pass. I'm also considering a smarter route: refitting the FFN outputs by least squares to compensate for pruning, which would need the exact FFN inputs, and I'm thinking about whether the attention probabilities over the 64 ids can supply those cheaply.

I'm trying to see whether a probe model can read the attention output from each layer in one pass, but reproducing attention in numpy would just be reimplementing the forward pass, which I want to avoid. Building one probe per layer would mean around ten passes, which is too many for the remaining budget, and I can't duplicate layers since each original layer can appear only once in the ordered subset.

I'm considering dropping some middle layers instead of shrinking the width, testing a pattern like skipping every third layer to reach about six or seven layers while keeping the full FFN size. If the accuracy holds near the M9 level, that would be a much cheaper route, though the reserved-dimension cuts in layers 2 through 10 and the carry heads at 2, 6, and 10 complicate which layers can be removed.

I'm weighing a layer-dropping experiment instead of shrinking width, keeping layers 0 through 4 plus 6, 8, and 10 to see how much accuracy an eight-layer subset loses at each depth. The alternative pattern, keeping 0, 1, 3, 5, 7, 9, and 10, would show whether early or late drops hurt more, though it would be harder to tell which removal caused any damage.

I'm considering dropping layers 4, 7, 10, and 11 and keeping 0–3, 5, 6, 8, 9, to see how well the drops hold up at each depth. Layers 0–3 match the base model exactly, so the first real change shows up at depth 5, where I'd compare against the base features. I'm also weighing whether to test concrete candidate stacks like a six-layer design directly, since literature on dropping top layers suggests some depths hold up better than others.

I'm going to try dropping layers 4, 6, and 8 to get a seven-layer stack, checking the mean features at each depth to see whether it stays close to the base model's kernel quality. If it does, the cost falls to around 81M, though that's still a big model. Dropping more layers would bring it to about 74M, but the savings from the deep path look limited, around 15 to 30 percent, compared with the shallow path's roughly 70 percent.

I'm weighing shallow approaches now, since dropping to a few layers saves far more than trimming depth does. The base kernel on the first layer already gets about 97.1%, against roughly 98% for the deep model, so the remaining gap is under a point. I want to see what attention pooling could add over those shallow features, starting with the token-level data from the pass_4 run.

I'm checking whether the first-layer token states could give a cheaper shallow readout, but reproducing the embedding step in numpy would just be re-running the forward pass, so I'm leaning away from that. The token states I can read directly from the existing passes already cover that ground.

I'm considering whether the 97.1% from the first-layer kernel is really the ceiling for anything built from those features, since a different nonlinear classifier might squeeze out more. To go past it, I'd need extra information beyond the first-layer mean, such as attention-weighted means or second-moment pooling over the token states.

I'm checking whether the first layer's FFN could build quadratic-like features per token, since GELU pairs of offset projections approximate second-order terms, which mean pooling would then average into a second-moment readout. The catch is that the residual stream already carries the raw token state, so any added FFN outputs would mix with it rather than sit cleanly alongside it.

I'm checking whether the layer-0 FFN could supply the per-token nonlinear features I'm after, since a mean over its outputs would add a second-moment-like readout on top of the first-layer mean. The difficulty is that the residual stream already carries the raw token state, so the FFN contribution would blend into it rather than sit separately.

I'm weighing whether attention pooling that emphasizes the strongest evidence, rather than a plain mean, could lift the shallow readout past the first-layer kernel ceiling. I'd rather test that with one pass than estimate it from the small token sample, since the threshold may be less tight than I assumed.

Checking the expected error counts at 97.8% CV against the test set, the margin looks like a coin flip. I'd need CV around 98% to be reasonably safe, which points back toward the deeper kernel path, though it only saves about 13% of parameters.

I'm comparing the kernel accuracies for smaller layer subsets, where j=6 through 8 all fall short of the M1+M9 level near 98.2%. That means I need the cutoff at j=9 or later, which pushes the layer count to at least ten. Checking the parameter counts across FFN widths from 1024 to 3072, the range runs from about 64M to 95M, and I'm still hoping layer dropping or FFN pruning can bring that down without losing quality.

I'm looking at whether pruning the FFN neurons could cut the model down without much loss, since the readout gets refit afterward. Published post-training pruning results suggest BERT can keep most of its accuracy under a FLOPs budget, though my proxy-based importance test may have undersold it.

I'm weighing whether pruning the FFN neurons or dropping layers could get the deep path under budget without losing too much accuracy, since the m=768 proxy test cost about three points. I'm also considering multi-block kernels that combine shallow and mid-depth means, like M0+M4+M8, to see whether they can match the M1+M9 level near 98.2% with fewer layers. With roughly 14 test passes left, I'll need to pick the candidates carefully.

I'm planning to run more numpy analysis on the existing base data to find the best multi-block kernel combinations for each maximum depth. The idea is to pair the full pooled features at depth j with low-dimensional carried scores from earlier layers, and the open question is how to fit those carried projections without leaking information into the cross-validation.

I'm now comparing a joint kernel that adds a quadratic term on the top layer's features to a linear term on earlier ones. Because the kernel solution is linear in those earlier features, their contribution can be carried as a small set of 8-dimensional scores, so I want to test how much that combination adds over the top-layer kernel alone.

I'm now checking whether the readout can be a kernel on the deepest pooled features plus a linear term on the earlier layer means, and I'll compute the cross-validation for depths 6 through 12 with those linear add-ons.



Deep features are the only route that reaches the roughly 98% CV needed to clear the 97.8% pass line, since a shallow model at 97.2% would likely fail. I'm now computing the cheapest deep design, a quadratic kernel on the pooled deepest mean plus linear terms from earlier layers, and its CV at each depth.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\ndef nz(X):\n    Z=(X-X.mean(0))/(X.std(0)+1e-6); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef zsc(X): return (X-X.mean(0))/(X.std(0)+1e-6)/np.sqrt(X.shape[1])\ndef kcv(K, lam=1e-2, seeds=(0,1,2)):\n    out=[]\n    for s in seeds:\n        f=folds(y,5,s); S=np.zeros((len(y),8))\n        for i in range(5):\n            tr=f!=i; te=f==i\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n        out.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(out)),4)\nq=lambda C: C+0.5*C**2\nGn={L: nz(M(L))@nz(M(L)).T for L in range(13)}\nGl={L: zsc(M(L))@zsc(M(L)).T for L in range(13)}\nfor j in range(4,13):\n    r=[kcv(q(Gn[j]))]\n    for c in [0.3,1.0]:\n        r.append(kcv(q(Gn[j])+c*Gl[1]))\n    r.append(kcv(q(Gn[j])+0.5*(Gl[1]+Gl[j//2])))\n    print(j, 'kern Mj', r[0], ' +lin M1 c.3/1', r[1], r[2], ' +lin M1,M_j/2', r[3])\n"
}
```


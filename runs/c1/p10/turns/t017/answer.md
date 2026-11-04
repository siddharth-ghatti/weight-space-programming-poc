## reasoning summary

I'm weighing the accuracy numbers against the cost of each model: the M0+M9 kernel is at 98.5%, while the M1 kernel is at 97.6% with gamma 0.8. The main worry is whether a k=2 model with about 97.5% cross-validation is safe enough, given that a model with around 97.8% CV tested at 98.31%.

I'm checking whether more seeds would shrink the cross-validation noise, since at 1496 samples the standard error is about 0.4%, which means a 0.3% gap is within the noise. Given that failing the threshold would be costly, I'm aiming for a model with at least about 98.2% CV and then minimizing parameters within that. The candidates that clear it are the M9 kernel, the M0+M9 kernel, and M10, all of which need layers 0 through 8 plus a pooling layer, so at least 10 layers unless the pooling can be merged into the last real layer.

I'm now looking at whether pooling could be folded into the existing layers instead of adding a separate layer, since that would save a layer. One option is to use the feed-forward units of the pooling layer as extra kernel features, which could be useful if they can pass class scores through the pooler.

I'm working out whether pooling could be folded into the feed-forward units of the pooling layer, so those units act as extra kernel features that feed into the classifier alongside the pooler's own features.

I'm weighing pruning the feed-forward units in layers 0 through 8 to shrink the model, though a pruned model changes the representations and would need its activations recomputed, which is costly given the read limit of 1024 ids per pass. With eight passes left, I can afford only about three candidate evaluations plus a final run.

I'm considering whether the pooling layer's output can be kept identical to the base model's, so that the existing reads on layers 0 through 8 still apply and I don't have to re-read activations after pruning. If the earlier layers stay intact, the features going into the pooling layer match the base exactly, and only the pooling layer itself needs to change.

I'm considering whether to prune the feed-forward units in the earlier layers to save parameters, which would be closed-form: keep a subset of units and refit the output projection to reproduce the original feed-forward outputs over the token activations. The catch is that it needs token-level activations, which means more read passes, and I have limited passes left.

I'm considering whether a closed-form ridge fit on the read activations could replace pruning, since I can't get the FFN intermediate values directly from the reads I have. The reads only expose the layer hidden states, token hidden states, attention maps, and logits, so any approach that depends on FFN activations would mean reimplementing the attention computation, which doesn't seem worth it. I'm now weighing what the available reads can actually support for fitting a compact model.

I'm now weighing whether pruning the feed-forward units is worth it, since fitting a closed-form replacement would need intermediate activations that the available reads don't expose. Without them, any pruning would have to rely on weight-based criteria, and its accuracy could only be checked through two-pass reads per evaluation. I'm also comparing the parameter counts of a few configurations, starting with the k=2 model at about 29.17M.

I'm comparing the parameter counts against accuracy: the k=1 configurations at roughly 26.8 to 31.5M land around 97.0 to 97.3%, while the k=10 model at 95.3M reaches about 98.3%. The gap between the two extremes is large in parameters but only a percentage point or so in accuracy, so I'm looking for something that sits between them.

I'm now considering whether a one-layer model could be made more accurate, since its attention can be written as a fixed weighting over the token embeddings: each head scores every token against a constant query from the CLS position, then sums the value vectors.

I'm considering whether a one-layer model could gain accuracy through smarter pooling, since each head's attention over the token embeddings is a fixed weighting from the constant CLS query, so keyword-presence and first-word features could carry more signal than a plain bag of embeddings. Fitting those weights in closed form would need token-level embeddings for the practice sentences, which is the next thing I'd have to work out.

Checking whether the embedding lookup counts as reimplementing the forward pass, and whether I can get token-level embeddings without reading them all through the limited passes. The weight matrices themselves can still be used directly, for example to pick key directions that favour particular vocabulary items.

I'm checking whether tokenizing the practice sentences and computing token-count statistics counts as building features outside the model, and whether computing embedding LayerNorm per token amounts to reimplementing the forward pass. I'd rather stay within what the reads expose, so I'm considering whether a one-layer model can improve its pooling using only the weight matrices.

I'm weighing a middle path: pruning the deeper layers' feed-forward units to about m=512 or m=1024 could keep the parameter count near the 52.8M of the k=4 model, but the k=4 features only reach roughly the same accuracy as layer 1, so it's not an obvious win.

I'm weighing a middle option: pruning the deeper feed-forward units to around m=512 would bring a 10-layer model down to about 56M parameters, though it's unclear whether accuracy would hold near 98%. The shallow k=1 model is much smaller at 26.8M but only reaches about 97.2%, and a seven-layer version at m=512 (about 46.5M) would land around 97.8%. Neither of the smaller options looks accurate enough to be safe, so I'm leaning toward the deeper path despite its size.

I'm thinking through whether a deep model is necessary, since the CV-to-test gap suggests a shallower model would probably land just under the target. The 98.3% configuration is the safer bet, so the remaining question is how to shrink it while keeping the deep features, maybe by letting the top prefix layer handle the pooling directly.

I'm testing whether combining an early layer's features with the layer 8 output could hold the accuracy up, and I'm planning to check pairs like M0 with M5 through M8, plus M1 with M8. I'm also looking at whether the carry heads from layers 2, 6, and 10 could feed in as extra ridge scores alongside a kernel on the final-layer mean.

I'm sketching how the pooling layer could work: uniform attention across all heads, so the CLS position just averages the token states, with a large scale so the CLS's own state gets diluted. Then I'd zero out the feed-forward output so the second layer norm sees only that averaged signal.

I'm considering using the 3072 feed-forward units in the pooling layer as extra kernel features over the pooled vector, with the ridge-combined class scores written into reserved dimensions for the pooler to read. The open question is whether a GELU unit with a large negative bias can act as a usable exponential-style kernel, since GELU's tail falls off differently from exp.

I'm checking whether GELU units with random centers can work as kernel features, since their positive regime is nearly linear. I also need to check the LayerNorm step: the reserved class-score dimensions get scaled by the per-sample standard deviation, which keeps the argmax intact on that path but changes how it mixes with the pooler's output.

I'm weighing whether computing the pooling layer's feed-forward output in numpy on the read pooled vector counts as reimplementing a model sublayer, or whether it's just the same kind of head-fitting I'd do for the classifier anyway.

I'm now weighing whether to fit the pooling head from read activations rather than simulating the forward pass, which would keep it within the rules. The idea is to zero the feed-forward output so the final CLS state is just the normalized pooled signal, then fit a simple pooler kernel on top of it.

I'm weighing whether a single kept layer could replace the deep stack, since a model around 26.8M parameters would be a large saving over the roughly 95M of the deep version. Before I work through that shallow option, I'll sketch how a 768-unit pooler fitted on the final CLS vectors might combine with an early layer's features.

I'm working out how a single kept layer could compute attention for the CLS query, since each head's weights over the token embeddings reduce to an exponential of a free vector dotted with each token's state. The value and output projections are free linear maps, so the question is whether one layer's attention can carry the pooling.

I'm working through the single-layer attention case: the CLS query's weight on each token reduces to a softmax over a free vector dotted with that token's state. With uniform weights this is just mean pooling, which gives about 97.3% on its own. I'm now considering whether a non-uniform, learned weighting over tokens could beat that baseline.

Checking whether computing the embedding output counts as running part of the model in the sandbox. The rules forbid simulating the forward pass, so the first layer's output is off-limits there. Token-level reads are capped at 64 ids per pass, which leaves too few passes to fit anything at the token level. That means for the single-layer case I only have mean-level reads to work with.

I'm considering whether a single kept layer could do the pooling through attention, but the attention weights would need token-level data to fit, and I can't get enough of that. Position-restricted pooling is an option in principle, though I can't evaluate how the position and word components mix after layer norm without it. A single layer looks too risky to pursue.

I'm weighing a k=2 setup next, with pairing M1 alone or with M0 giving roughly 97.5–97.6%, but attention pooling there looks risky. The bigger idea is that the pooling layer's 3072 FFN units could act as kernel centers over the pooled CLS vector, which might approximate a full kernel ridge classifier over all 1496 training points.

I'm considering whether the pooling layer's GELU units could serve as kernel features over the pooled vector, with each unit centered on a training point. GELU isn't exponential, but a dot-product kernel of the form GELU(a·z·c + b) could still work as ridge features across all 1496 training points, and should land near a reasonable kernel ridge classifier.

I'm going to check whether GELU features with a dot-product form, using every training point as a center, can do the job of a kernel ridge classifier, and compare how M1 and M9 score under that. The idea is that the reserved class-score dimensions in the FFN output would carry the ridge combination, while the LayerNorm scaling per sample would need to be accounted for.

I'm checking whether the LayerNorm after the feed-forward step stays close to constant per sample if the first normalization uses unit gain and zero bias on the non-reserved dimensions, since that would keep the class-score scaling roughly fixed.

I'm noting that the LayerNorm scaling keeps the argmax of the feed-forward scores intact, so the prediction would just be that argmax, with the scores effectively acting as a kernel ridge classifier. That still leaves the features as the real bottleneck, not the decoding step.

I'm checking whether the shallow models really sit around 97.5% while the deep ones reach 98.3–98.5%, so I'll rerun repeated folds for the M1 and M0+M1 kernel ridge candidates alongside M6, M8, and M9. I also want to see whether a modified layer 0 attention could give the shallow features a boost.

I'm considering whether a simpler approach could work: a single layer whose attention pools position-based features, with the FFN acting per token. The catch is that without a further layer, the pooler only reads the CLS token, so such a model could only pool roughly linear functions of the embeddings. A second layer would let the model pool nonlinear per-token features, which is what the M1 variant gets at, so I'm weighing whether moderate depth with pruned feed-forward layers might be a better tradeoff.

I'm now weighing a non-prefix layer subset, dropping a few middle layers the way BERT tolerates through its residual stream. Candidates like keeping layers 0 through 3 plus 5, 7, and 8 might keep most of the gain from depth at lower cost. I still need repeated CV to check whether the jump at layer 9 holds.

I'm weighing a plan to build a deep candidate with layers 0 through 8 plus a pooling layer, at roughly 95M parameters. The point is to stay safely above the 98.3% CV level. I'd then try trimming it, either by pruning the feed-forward width or by dropping some middle layers, since fewer exported parameters is better among passing models.

Pruning FFN units looks hard to judge without activation statistics, since the attention sublayer's output isn't available to compute them. So I'll have to test any pruning empirically through reads, one evaluation per candidate.

I'm weighing whether to prune layers and feed-forward units from the deep model, since a single read pass on a specific pruned configuration would show how much accuracy survives. Dropping units by weight norms alone is a guess, so I'd want an empirical check for the chosen experiment.

I'm planning to keep layers 0 through 8 and reuse layer 11 as the pooler, since its LayerNorm will normalize whatever input it gets. A read pass on that edited model should show whether the mean hidden states at each layer still match the base model's, and it would also give me the final CLS vector to fit the kernel head on.

I'm planning a read-only pass on a pruned variant, with layer 11's FFN output already zeroed so I can keep whatever width I choose. The next step is to test m=1536 by weight magnitude, then fit the kernel head on the resulting CLS vectors and see how much accuracy survives.

I'm planning a read pass on the edited model with pruned FFN width, pulling mean hidden states and the final CLS vector for the first 1024 samples. I'll fit the kernel head with cross-validation on that subset, and run the same CV on the base features so the two are directly comparable. The deeper layers' attention should still carry most of the contextual signal even if the width shrinks.

I'm considering whether the residual stream would keep the bag-of-words signal even with the FFNs stripped out, since the kernel result suggests a floor near 97%. Dropping every FFN would cut the model to about 48M parameters, while keeping a width of 768 would land near 60M, so narrowing the width is less of a saving than removing whole layers.

I'm tabulating the parameter counts for different depths and widths, and the pattern is that width costs scale with the number of layers. For the deepest setting with 10 layers, going from no feed-forward units to 3072 raises the total from about 48M to 95M. The shallower options are cheaper, and a single layer at full width would still come in around 31.5M, which makes it look like a kernel classifier on the first layer's features could be a serious candidate.

I'm checking whether kernel ridge on M0 alone, or on M0 plus M9, lands near the original 97.8% CV with tuned gamma. If M0 sits around 97.3%, a one-layer model would expect about 97.7% on test, which is roughly a coin flip against the threshold, so I'm leaning against it. I'm still curious whether per-token nonlinear features from a single layer with attention pooling could close that gap.

I'm thinking about whether a one-layer model could get the same benefit: the FFN would run per token and the pooling would happen in the next layer, so the features are nonlinear before they get averaged. Mean-pooling random token-level features like that amounts to a kernel mean embedding over the tokens, which is a bag-of-words model with a kernel on the tokens.

Pondering whether the mean of nonlinear token features beats the mean of raw embeddings: the embeddings are already 768-dimensional, so a plain mean carries a fair amount of information, and a kernel on that mean adds interactions on top. The layer-0 features reached about 96% with a linear head and 97.5% with a kernel, which I'm measuring against the original threshold.

I think the shallow route tops out around 97.5%, while the deep route gets to 98.3–98.5%. Layer 9 looks special, so I want repeated CV on layers 5 through 10, plus some combinations with layer 0, before I decide.



Shallow layers reach about 97.5% CV and L9 hits 98.3%, against the original model's 97.8%. I'm running repeated 3-seed CV across layers 5–10 to firm up those numbers, since the 1-sigma noise on 1496 items is about 0.4%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef zs(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef krr_multi(blocks, y, gammas, lams, seeds=(0,1,2), k=5):\n    n=len(y); res={(g,l):[] for g in gammas for l in lams}\n    for s in seeds:\n        S={(g,l):np.zeros((n,8)) for g in gammas for l in lams}\n        for te in folds(n,k,s,y):\n            tr=np.setdiff1d(np.arange(n),te)\n            Z=np.concatenate([zs(X,tr) for X in blocks],1)/np.sqrt(len(blocks)); sq=(Z**2).sum(1)\n            D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y[tr]]-1/8\n            for g in gammas:\n                K=np.exp(-g*D2); w,V=np.linalg.eigh(K[np.ix_(tr,tr)]); R=V.T@T; Kte=K[np.ix_(te,tr)]\n                for l in lams: S[(g,l)][te]=Kte@(V@(R/(w+l)[:,None]))\n        for key in S: res[key].append((S[key].argmax(1)==y).mean())\n    return {k_:round(np.mean(v),4) for k_,v in res.items()}\nout={}\nfor name,bl in [('M0',[M[0]]),('M1',[M[1]]),('M5',[M[5]]),('M6',[M[6]]),('M7',[M[7]]),('M8',[M[8]]),('M9',[M[9]]),('M10',[M[10]]),('M0+M8',[M[0],M[8]]),('M0+M9',[M[0],M[9]]),('M1+M9',[M[1],M[9]])]:\n    r=krr_multi(bl,y,(0.2,0.4,0.8),(0.003,0.03))\n    best=max(r.items(),key=lambda t:t[1]); out[name]=r\n    print(name,'best',best, 'all',list(r.values()))\n"
}
```


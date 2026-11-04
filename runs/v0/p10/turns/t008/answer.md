## reasoning summary

I'm noting that many errors are really ambiguous or semantic, like treating "is a covered loss" as an exclusion, and that some could be fixed by stronger models. Two ways to improve: combine the layer-0 and deep layers for a small gain of about 1%, or use the full 1496 examples for training. I'm now sketching a multi-layer combination I could actually implement, starting with an initial idea.

I'm weighing a multi-layer pooling idea that mixes the CLS and mean representations at layer 11, since the CLS token alone gets about 90.8% and the mean could complement it. A sum-based mix would stay at 768 dimensions and lose the benefit of concatenating, so that version would only be a partial gain. I'm also considering splitting attention heads at the final layer so different heads pool differently, which might capture different features but seems complicated.

I'm weighing whether the final CLS vector could hold compressed pieces from two layers instead of a plain sum. The ridge classifier only needs 8 outputs, so a few discriminative directions from each layer might be enough to fit both in 768 dimensions.

I'm checking whether the final CLS vector can carry both layers' contributions by linearity: if the ridge logits are just linear in the pooled means, the two could be summed into the same 768 dimensions as long as each projection lands on the classifier's 8 output directions.

I'm now looking at whether the final CLS could absorb both layers linearly, since the ridge logits are linear and the pooled features could map onto the same 8-dimensional output subspace. Working from the last layer alone, the heads only see layer-11 tokens, so any earlier-layer information would have to be approximated from those tokens, and I'd be relying on the ridge fit over mean(h11) to carry it.

I'm considering whether the earlier layers could be pooled alongside the final one, or whether the concatenated means of two layers would be worth checking in cross-validation. I'm also wondering if a repeated multi-fold CV would give more reliable comparisons, since a single L0 mean with strong regularization already looks reasonable at about 0.95.

I'm considering whether a pooling layer could embed the per-token L0 ridge contributions into a small residual subspace, but post-LN renormalization and the attention and FFN writes at each sublayer would distort that subspace, so it wouldn't survive intact. I'll set that aside and accept roughly 96% from single-layer pooling, making sure the fit is solid using all 1496 examples, and then look at improvements if there's time.

I'm sketching a variant where the final CLS token attends uniformly to the other tokens while at an earlier layer, using the query weights to make CLS behave differently from ordinary tokens, so that only the CLS position picks up the mean of an earlier layer.

I'm working out whether the CLS query could be made to attend uniformly while the other tokens keep their normal attention, by having the query for the CLS position shrink toward zero so its scores become constant across keys.

I'm asking whether the CLS query could be neutralized by projecting out the near-constant CLS direction, so its attention goes uniform while other tokens keep theirs. The catch is that the CLS residual varies a little across sentences, so the leftover query may not be small enough to flatten the scores.

I'm now going to compare pooled features across several layers, including mean pooling from layers 8 to 12 and a concatenation of the CLS and mean at layer 11, using repeated cross-validation so the results are more reliable.

I'm looking at whether the attention output could be shaped to match the residual, by setting the output projection so the mean of the values lands in a chosen subspace alongside the CLS state.

I'm considering whether the residual x_cls could be cleared out, but it carries real information at layer 11, so it can't simply be removed and a large scaling constant would be needed to overwhelm it. I'll settle on pooling h11 at layer index 11 with no other edits, fitting the ridge on all 1496 examples for roughly 96%, and maybe check whether pooling h9 with pass-through layers does a bit better.

I'm now weighing whether a mean-pooling approach across several layers could beat single-layer pooling, since the pooler's random-feature or FFN-based alternatives don't look like they'd add much.

I'm considering a variant where the CLS token's state is set after pooling at an earlier layer, so it no longer influences other tokens in later layers. The difficulty is finding a direction that separates CLS from ordinary tokens in the keys.

I'm considering whether the CLS state after an early layer could be made irrelevant to other tokens, but the difficulty is that its keys and values change once its state changes, which affects how the other tokens attend to it.

I'm weighing whether to pool the final layer's CLS as a scaled mean, since attention to CLS in middle layers might be modest; the idea is to read the mean of layer 11 directly in the edited model and measure accuracy rather than guess. I'm also considering an early-pool variant at layer 0, but pooling there would change what CLS gets from every layer after it, which looks risky.

I'm comparing ridge fits on concatenated layer means, such as layers 9 and 11 or 9 and 12, against L0 pooling, which seems to carry something distinctive. I'm also weighing whether pooling the embeddings at layer 0 into the CLS would reach every token, since that would affect the whole network rather than just the CLS position.

I'm planning the next passes: first an edited-model read on all 1496 examples to get pooled features, then a cross-validated comparison before deciding whether the extra budget is worth spending on improvements.

I'm weighing a design where the attention pooling happens at an earlier layer, around layers 8 or 9, with all heads, to see whether that keeps more of the lexical signal than pooling only at the final layer.

I'm checking the uniform-attention case: if every head at layer a attends evenly, every token receives the same mean of h_a, so the later layers would just see identical tokens and the result would match pooling at a. With a small scale instead, each token keeps its identity but picks up a shared offset from the mean, and I still need to work out how the remaining layers would handle that.

I'm considering whether a design where layer 10 reads out each token's lexical contribution could be pooled at layer 11, since the token identity should still be decodable from those states.

Next I'm checking whether the layer 0 signal is just a regularization effect, since concatenating layers with ridge changes the effective prior, and comparing the (0, 12) and (0, 9) concatenations against the deep-only pairs.

I'm now looking at a more careful comparison using repeated CV over many fold splits, covering concatenations like (0, a) and (a) alone. I'm also considering Design Q, which pools at layer 0 using only the CLS position: the CLS input at that layer is a fixed vector, so a head there could give its query a constant form and have the keys carry the lexical signal.

I'm working out a way to change only the CLS query in that layer-0 head, by subtracting its constant part along the CLS input direction so the rest of the tokens keep their original queries.

I'm working out how to remove only the CLS query's constant part at layer 0, so the other tokens' queries stay mostly intact. The catch is that the token embeddings are fairly correlated with the CLS input, so a direct subtraction would still disturb the other queries noticeably. I'm looking for a direction that matches the CLS input but is close to orthogonal to typical token states.

I'm checking whether the CLS query at layer 0 can be isolated by finding a direction that is nearly orthogonal to ordinary token embeddings, since the CLS input is fixed and known in advance.

I'm weighing whether the embedding-layer output can be computed directly in numpy, or whether I should stick to reads from the model, and I'll go with reads to stay on the safe side. The CLS state at layer 0 is a fixed vector I can take from the read, and the token states at that layer can come from a read as well, so I can estimate the leakage of a CLS-only query direction before deciding how to build the head's value path.

I'm working through how CLS would carry the mean of the embedding values through this head, since the original output projection keeps other tokens unchanged. The catch is that CLS's residual would shift, which then changes what other tokens read from it in the later layers, and the information would have to survive eleven more layers.

I'm settling on single-layer pooling and will pick the layer through careful cross-validation. I may also test whether excluding CLS and SEP from the attention weighting helps, though I'm not sure it will. Before that, I want to check a Design R that keeps everything in the final layers and uses the varied attention patterns across heads at deeper layers. Mean pooling at layers 8 through 11 already reaches about 96%, and concatenating (8, 12) or (9, 12) gives small gains, so I'm wondering whether max-pooling or attention-weighted variants could push further.

I'm weighing whether pooling the attention output at an earlier layer, with all heads, would keep more of the lexical signal, and I'm considering how the later layers would treat tokens that share a common offset. I'm also reasoning that the sentence-structure classes and the register confusions in the errors need more model capacity than I have, so I'll focus on comparing single-layer pools carefully.

I'm comparing the single-layer pools at indices 8 through 11, since the differences look small, and I'll run a repeated 5-fold CV over ten splits to see which one holds up. I'm also going to try a shrinkage LDA as an alternative classifier.



Most remaining errors are subtle cases rather than missing keywords, so I'll compare layers with repeated cross-validation before picking one to mean-pool, and try a shrinkage-LDA variant too.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in range(13)}\ndef cv_acc(X, y, lam, reps=6, mode='ridge'):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            if mode=='ridge':\n                Y = np.eye(8)[y[tr]]\n                Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n                s = Xv@Wt\n            else: # shrinkage LDA\n                means = np.stack([Xt[y[tr]==k].mean(0) for k in range(8)])\n                R = Xt - means[y[tr]]\n                S = R.T@R/len(tr); S = (1-lam)*S + lam*np.trace(S)/d*np.eye(d)\n                Wt = np.linalg.solve(S, means.T)\n                s = Xv@Wt - 0.5*np.sum(means.T*Wt,0)\n            c += (s.argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nfor l in [6,7,8,9,10,11,12]:\n    print('L',l, 'ridge', [cv_acc(M[l],y,lam) for lam in [50,100,200,400]], 'lda', [cv_acc(M[l],y,s,mode='lda') for s in [0.1,0.3,0.5]])\n"
}
```


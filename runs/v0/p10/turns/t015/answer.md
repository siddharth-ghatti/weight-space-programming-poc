## reasoning summary

The CLS token barely changes across the early layers, with its variance staying tiny compared to ordinary tokens, so it acts like a fixed sink there. That means any sentence-dependent shift in CLS at those layers would show up as new variation for tokens that attend to it. I'm considering adding a small perturbation to see how that propagates.

I'm weighing whether to inject a small perturbation into the CLS token, sized near its natural layer-1 variation, to carry bag-of-words information forward. The open question is whether that signal survives propagation through the layers, since the Jacobian depends on context, and the natural CLS variation at layer 9 already carries some of that information.

I'm thinking about whether a shared bias added to every token could carry the sentence information through the mean features, since a linear ridge might pick it up if it stays in unused directions. That's messy, though, so I'm looking for a cleaner route: a residual-stream subspace that the later layers' weights largely ignore, so a component placed there would pass through untouched.

I'm checking whether a component placed in a direction the later layers ignore would survive the LayerNorm steps. Each LN rescales it by a per-token factor and applies gamma elementwise, so the readout would be the original magnitude times a slightly varying scale, with some distortion from gamma.

I'm checking whether a direction in the residual stream could be nearly ignored by the later attention and FFN weights, which would let a component there pass through layers mostly unchanged apart from LayerNorm scaling. That seems hard to guarantee for full-rank matrices, though BERT's low-usage directions might make it possible. The approach looks too complex, so I'm reconsidering whether it's worth pursuing.

I'm also weighing whether pooling at different layers could be spread across heads, but tokens that attend uniformly at late layers would mix that signal into every position, so it doesn't stay clean.

I'm looking at whether the CLS token could act as a detector in the late layers: tokens rarely attend to it there, so a CLS-specific query change would do little damage. I'd need a direction u that picks out the CLS state from the token distribution, which I might estimate from the token covariance with some regularization.

I'm looking at whether the sacrifice could be made per layer instead: at layers 5 to 8, one head could be given up for every token, so its attention no longer has to single out the CLS position.

I'm now considering whether pruning a head at one layer would let it act as a uniform pooler, though its output would then be mixed into every token's stream. Since many BERT heads can be removed with little loss, giving up a few heads might be an acceptable cost, but the pooled signal would still blend with the ordinary mean features.

I'm weighing a two-stream design where the CLS-only writes stay clean, but injecting bag-of-words into CLS at layer 0 would change the attention outputs of layers 2 to 4 for every token, which would likely hurt the mean features at layer 9. I'm reasoning that with seven passes available, one or two experiments are affordable before the final two passes.

I'm still turning over a way to carry the bag-of-words information in without disturbing the other tokens, and I'm considering injecting it at a later layer where the CLS token's own keys and values are what matter.

I'm considering letting the CLS token's own query pick out token identities through a content-dependent softmax, so a max-like pooling could capture features the plain mean misses. I'm also weighing whether a head that scores tokens with a linear direction could give a nonlinear presence signal for a given class.

I'm considering whether the max-style pooling could come from per-token class scores, taking the maximum over tokens so the classifier gets features the plain mean misses. The catch is that I only have token-level data for 64 texts, which is too few to measure any gain from it, so I'm inclined to drop the idea rather than overengineer it.

I'm weighing the final score against the effort: the gain from a different pooling layer is around a point, so it's worth trying but not risky. I'll build the v1 version now with a solid baseline, then explore further if I have passes to spare. For the design, I'm comparing pooling at layer 8 or 9, since those means score about 0.96, against layer 11 at about 0.954, and the CV numbers favor layer 8.

I'm settling on pooling at layer index 8, since its CV score beats the later layers, and leaving layers 9 through 11 as pass-through. I'm also weighing whether pairing two deep layers through CLS routing might add half a point, and whether injecting information into CLS at layer 0 could work if its state were restored before the early layers attend to it.

I'm weighing whether to shift the representation at layer 0 with one head made uniform across all tokens, but I'll drop that. Instead I'm testing whether the gain from pooling at layer 0 comes from a few lexical cues, like negations or "premium," that deeper layers blur, by checking a ridge model on the mean-pooled layer 9 features combined with a top-k PCA slice of the layer 0 mean.

I'm considering whether a common shift written into every token at layer 0 could carry the bag-of-words logits without affecting the per-token content, since that shift would be identical across positions.

I'm checking whether a common shift d added to every token at layer 0 would be nearly harmless: keys and values would move by the same vector, so the attention weights and mixed output barely change, but queries, the FFN, and LayerNorm would still be affected, so it only stays small if d is small.

I'm weighing whether a common shift added to every token at layer 0 could carry the bag-of-words signal, since the same vector across positions would be replicated and averaged by mean pooling. I'm checking whether, if that shift is projected from the BoW logits, the deep features would absorb it cleanly enough for a ridge readout.

I'm considering a common shift d added to every token at layer 0, since mean pooling would average that identical signal across positions. The idea is to make d small in directions that are harmless to keys and values, and then check whether the deeper features can still be decoded cleanly. The catch is that LayerNorm rescales the tokens when d is large, so the shift only stays safe if it remains modest.

I'm checking whether a small common shift added to every token at layer 0 would stay harmless: the shift cancels out of the keys and values, but it still perturbs queries, the FFN, and LayerNorm, so the imprint on the mean-pooled features has to stay modest to be decodable.

I'm checking the imprint size against the top principal components of the layer-9 means, since a fixed-norm shift along a few directions would stand out against the smaller ones. Then I'm planning one pass to read the edited model's means at every layer and run cross-validation, adopting the change only if it clearly beats the 0.96 baseline. I'm still trying to work out how to write the shift into every token at layer 0.

I'm considering making one layer-0 head uniform across all tokens, which would have it write the same mean value vector into every position. Head 0 already looks close to uniform, so the change might cost little, and I'm planning to redefine its value projection to carry the bag-of-words signal.

I'm considering using the value projection of this layer-0 head to carry the bag-of-words signal, by adding a small rank-8 component to its value vectors. The open question is whether an extra component in the head's value space can be routed through the existing output projection so it lands in useful residual directions.

I'm weighing whether making one layer-0 head uniform across tokens could still carry the bag-of-words signal through its value projection, since its original output is already close to a plain average. The open question is whether the extra value component can be routed through the output projection into residual directions that survive, and I'm treating that as a closed-form design problem rather than something to settle by experiment yet.

I'm wondering if injecting a shared shift at layer 0 could work the way a sentence-type embedding does, since BERT might tolerate that. Another option is to put the BoW signal in a uniform-attention head at a later layer, but then it's just the mean of layer 8, so it adds nothing new.

I'm now checking how many bag-of-words dimensions are actually needed, by testing a ridge readout on the layer-9 means combined with either a PCA slice or the logits of the layer-0 mean. If eight logit dimensions are enough, I'd inject just those through the shared shift, which would avoid any extra routing.

I'm settling on a PCA slice of the layer-0 mean rather than supervised BoW logits, since the logits fit on the same practice items would make the final ridge overtrust them. I still need to pick how many components k to test.



[CLS] is nearly constant across sentences in early layers and acts as an attention sink, so injecting sentence information there early would disturb other tokens. I'll try a small shared bag-of-words vector added to every token at layer 0 instead, first checking how many layer-0 dimensions are needed on top of the deep mean.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in [0,1,8,9]}\ndef cv_acc(X, y, lam, reps=4):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nX0 = M[0]-M[0].mean(0); U,S,Vt = np.linalg.svd(X0, full_matrices=False)\nfor k in [8,16,32,64,128,256]:\n    Z = X0@Vt[:k].T\n    print('L0 pca',k, 'alone', cv_acc(Z,y,30), ' +mean9', [cv_acc(np.concatenate([M[9],Z],1),y,lam) for lam in [200,400]])\n"
}
```


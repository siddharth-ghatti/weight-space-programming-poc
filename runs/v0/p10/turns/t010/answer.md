## reasoning summary

I'm checking whether pairing layer 0 or layer 1 with layer 11 beats layer 11 alone: the combined accuracy is about 97%, against 96% for the single layer, which is a gain of about 1.6%. Next I'm working out how to get the information from h0 or h1 into the final CLS token, since the last layer only receives h11 as input.

I'm weighing whether to route information from layer 0 or 1 into the final CLS token, since the last layer only sees layer 11's states. One option is to have CLS carry an early-layer signal through a subspace the later layers' key and value matrices ignore, though layer normalization and the attention to CLS from other tokens complicate that.

I'm now asking whether the layer 0 signal could reach the final CLS token without disturbing other tokens through the later layers' key and value matrices, perhaps by parking it in a near-null subspace. Since no exact null space exists, that seems fragile. I'm also considering whether an approximation of the layer 0 embedding could be recovered from layer 11 states at the end, but the MLM-head route doesn't work, since recovering token identity would need a full vocabulary softmax.

I'm testing whether the layer 0 mean carries lexical identity, since a ridge fit on it alone reaches 0.95, so it works roughly as a bag-of-words model and adds to the deep features. Another idea is to have layer 0 apply a uniform shift to every token, so the mean-pooled final features pick up that information through the later layers without relying on the CLS token.

I'm now weighing whether the layer 0 signal could be carried in a way that touches the final layer's per-token computation. Since the layer 11 token states already encode the whole sentence, I'm considering modifying layer 10's feed-forward output so it yields a more lexical representation. The catch is that a linear readout of token identity from those states would already be captured by ridge on the mean, so the layer 0 gain has to come from somewhere else.

I'm wondering whether an FFN at layer 10 could act as nonlinear token detectors that recover the layer 0 information a linear readout from layer 11 misses. Without knowing which detectors would help, that's hard to pin down, so I'm considering a different route: making layer 0's attention for the CLS token alone do the mean-pooling, while keeping the other tokens protected.

I'm weighing whether to measure how much perturbing the CLS state at layer 0 disturbs the deep representations of the other tokens, by fitting a cross-validated ridge on the CLS and mean-pooled hidden states across layers. The final layer would then need to combine the CLS state, which carries the h0 mean, with the mean of layer 11, but a sum rather than a concatenation.

I'm considering splitting the final layer's heads so some attend to the mean of layer 11 and others attend only to the CLS token, with the CLS information projected into a subspace orthogonal to the mean-pooled one, and the residual shrunk by scaling the output projection.

I'm settling on a concat in disjoint subspaces: heads 6–11 would attend only to CLS by giving its key a strong marker direction, so the CLS state gets projected into a subspace separate from the mean-pooled one, with the residual scaled down.

I'm settling on routing: the final layer's residual is shrunk, and the layer 11 mean goes into one set of heads' subspace while the CLS state goes into a disjoint one, so the final CLS is a near-concatenation of both. I'm picking the directions for each subspace from the discriminative components of the mean-pooled layer 11 states and of the CLS state respectively, with 64 dimensions each likely enough for eight classes.

I'm looking at whether the final CLS token could carry the layer 0 mean in a subspace separate from the mean-pooled layer 11 features, with the residual shrunk so the two stay disjoint. Heads 6 to 11 would attend only to CLS while heads 0 to 5 pool over all tokens, and the open question is whether the CLS state actually carries that early information.

I'm going to design the early CLS trick: fix the layer 0 CLS input to a constant vector, then adjust the query weights so CLS attends uniformly while the other tokens' queries stay roughly unchanged.

Working through the algebra: with the query shifted so CLS attends uniformly, its attention output reduces to a linear map W_O·W_V applied to mean(h0), so the layer 0 information survives into the CLS state before layer normalization. The remaining question is how that signal shapes CLS after layers 1 through 10.

I'm now looking at whether the CLS state could be frozen through layers 1 to 10, but LayerNorm makes an exact freeze impossible without altering the other tokens. That pushes me toward moving the pooling step later, to around layer index 10, so the mean information gets injected closer to the final representation.

I'm considering whether the CLS token could carry the layer 0 mean through the ten layers in between, though the normal processing might wash out that information. A linear readout of the eight ridge logits from the final CLS state might survive partially. Another option is to encode those few numbers more robustly rather than the full mean.

I'm considering having CLS attend only to itself in layers 1 through 10, so its update becomes a fixed per-layer function of its own state, independent of the other tokens. The open question is how much of the layer 0 mean information would survive those layers.

I'm checking whether modifying CLS's query in layers 1 through 10 to attend only to itself would be feasible. The main worry is leakage: if other tokens attend heavily to CLS in some heads, changing CLS's key and value would shift their representations as well.

I'm checking whether making every token carry the bag-of-words signal would work, by adding a shared vector from uniform attention in one layer-0 head. That would avoid relying on a single CLS carrier, though the cost to the other tokens still needs weighing.

I'm dropping the idea of an early-layer CLS hack, since it would perturb other tokens too much for an expected gain of at most 1.5%. I'll keep the single-stream pooling and check whether weighted pooling that excludes CLS, SEP and punctuation, or pooling at layer 8 versus 9, does better. Splitting layer 11's heads so some do uniform pooling and others do different weighted pooling is the other option I'm weighing.

I'm weighing a position-based pooling head that attends to the first few tokens, since FAQ questions tend to open with similar words, and a max-like pooling head using a sharpened softmax over a class-relevant direction. Testing either would need per-token hidden states, which I only have for a limited number of texts.

I'm noting that the attention scores can be written as a linear function of each token's state, so a head can select tokens by an arbitrary direction. That would let one head act as a soft max-pool over the tokens most indicative of a given class, returning a projected value instead of a plain average.

I'm weighing a soft max-pool over tokens, where each class direction picks out the most indicative token. The attraction is that it could catch keyword-driven cues the plain mean washes out. The catch is that testing it would take several more passes, and the expected gain is still uncertain.

I'm planning the next pass: an edited model where some heads in layer 11 do uniform pooling and others attend by class direction. The classifier would still be fit on the subspace from the plain pooled features, so any extra features only get used if cross-validation supports them.

I'm weighing whether to pool at layer index 9 instead of 11, since its mean is slightly stronger. The catch is that pooling earlier means the later layers have to pass the CLS state through, and with uniform attention every token would collapse to the same vector, so I'd need to check how that interacts with the remaining layers.

I'm checking what happens if pooling moves to layer 9: with uniform attention every token collapses to the same vector, so the later layers would only see a function of that mean. Changing only the CLS query at layer 9 would leave the other tokens intact, but CLS would then influence them in layers 10 and 11 through attention, so combining layers 9 and 11 probably isn't worth much.

I'm weighing whether to pool at layer 9 or 11, since the mean is slightly stronger earlier, though I'd still need pass-through layers after it. For the final layer I'm considering splitting the heads: some doing uniform pooling of the layer-11 states, and others doing something different, like attention-based pooling that picks out tokens along a class direction.

I'm working out how to make the final layer's heads do uniform pooling: with V set to a 512-row projection, the concatenated output is just the mean of the layer-11 states, mapped back through W_O. Before building that, I want to check how much information is lost when the mean is projected onto its top PCA directions.

I'm weighing whether the soft max-pooling heads are worth their cost, since I can't test them without token-level reads, which cap out at 64 texts per pass. A simpler route might be the best pooling layer with pass-through and the remaining passes spent on verification. I'm also checking whether the experimental heads could be included in the same edited model so one pass serves both the final fit and the evaluation.

I'm going to see whether the pooler can stay nearly linear, so I'd set its weight to a small scale and check how far the LayerNorm outlier dimensions push the tanh.

I'm comparing the pooler's tanh saturation against a near-linear scaled identity, and I'm leaning toward the scaled identity since the base NSP pooler would lose information. Fitting the classifier on the actual pooled outputs keeps the result exact either way. I'm also reconsidering whether pooling at layer 8 or 9 is worth the extra pass-through layers.

I'm setting the layer's attention to uniform, so every token gets the same mean, with the value and output projections as identity-like maps and the feed-forward weights zeroed so the block passes through. Scaling the output by a large constant pushes each token toward the layer-normalized mean. The later layers then receive identical tokens throughout.

I'm setting the attention output and FFN weights in the later layers to zero, and both LayerNorms to identity, so each token passes through unchanged. Since the tokens differ slightly by the x_j/c term, I'm checking that this doesn't matter once the output projections are zeroed. The constant c still needs choosing so the mean dominates.

I'm checking how the mean-pooling constant c affects the residual: with c around 1000, the per-token term x_j/c is only about 0.1% of the mean, so it barely matters. Float32 should handle the larger magnitudes once LayerNorm normalizes them.

I'm settling on a design where heads 0-7 at layer 9 do uniform pooling, projected to 512 dimensions, and the remaining heads try something else. Before committing, I want to measure how much information is lost by reducing the mean of layer-9 states to 512 dimensions, and also check 384.

I'm asking whether the FFN of layer a-1 could be kept as a per-token nonlinear step before pooling, since that would supply the nonlinearity a plain mean lacks, even though it means giving up the original FFN's function in that layer.

I'm weighing a simpler route: pool at the best layer with pass-through and spend the remaining passes on verification. I'm also considering whether to repurpose some of the feed-forward units in the layer before pooling to act as per-token detectors, though that would change that layer's function. I'll set that aside for now and look at a design where layer 0 handles the CLS attention uniformly.

I'm planning a first version that pools at layer 9 (or 8) with all 12 heads doing uniform attention and the later layers passing through, which should give a solid baseline. After that, with the remaining passes, I'll try to carry the layer-0 information forward, even though that could cost a little quality in the deeper layers.

I'm checking how much the PCA reduction of the layer-9 mean costs in CV, and whether layer 8 or 9 is the better place to pool. I also want to see how constant the CLS hidden state is at layer 0.

I'm checking how constant the CLS hidden state is at layer 0 across different sentences.



Two-stream fusion would reach about 97% against about 96%, but I'm setting it aside as too complex and risky: I'll build a solid single-stream v1 first, then explore two-stream with spare passes. Next I'm checking whether mean-pooled features still hold up after projection to fewer dimensions, and whether the layer-0 [CLS] vector is constant.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in [0,8,9,11]}\nc0 = np.load('pass_2/cls_hidden_L0.npy'); print('cls L0 spread', np.abs(c0-c0[0]).max())\ndef cv_acc(X, y, lam, reps=4, std=True):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6 if std else np.ones(d)\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nfor l in [9]:\n    X = M[l]; Xc = X - X.mean(0)\n    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)\n    print('sv', S[:5], S[100], S[300], S[500])\n    for k in [64,128,256,384,512]:\n        Z = Xc@Vt[:k].T\n        print(l, k, [cv_acc(Z,y,lam) for lam in [50,200]], [cv_acc(Z,y,lam,std=False) for lam in [1,3,10]])\n# also per-dim magnitude of mean features (outliers)\nprint(np.abs(M[9]).max(0).argsort()[-5:], np.abs(M[9]).max())\n"
}
```


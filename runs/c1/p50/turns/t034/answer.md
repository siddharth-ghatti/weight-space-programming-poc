## reasoning summary

With several seeds, the M0 model scores 97.6% at 10 folds and 97.7% at 20, while M0M1 reaches 97.9 and 98.0. The original features with this kernel land at 98.2 and 98.4, against 98.0 and 98.2 for linear. So a k=1 model built on M0 plus the kernel would probably come in around 97.8%, which leaves roughly even odds of failing the threshold, and I'm weighing whether M0M1 with k=2 avoids that risk.

I'm weighing whether a model with k=2 on M0M1 avoids the threshold risk, since k=1 on M0 plus the kernel would land near 97.8%, roughly a coin flip. To lift a small model, I'm considering multi-head attention pooling: split the heads so some carry the uniform mean, compressed by PCA, and the rest carry class-evidence and position-based pools.

I'm picturing the position embeddings as a key direction: subtracting the average of the first few positions would pull attention toward the start of a sentence, where phrases like "Coverage shall not apply" differ from "We will pay." Each pool gives a different view, and the kernel head sits on top. Before settling on this, I'd want to run experiments on a one-layer model.

Next I'm working out how to get the FFN units, anchors, and readout installed on the final model. I'm also weighing a k=2 variant that keeps layer 0 at full width, with the pooling layer taking the anchors. The narrower width at m=1512 costs some fidelity, so I'll check that trade-off before committing.

Weighing k=1 against k=2: the M0M1 gain is small but real, about 0.25% for roughly 16% more parameters. Attention pools might recover some of that gap without the extra layer, though I'm not sure they will.

I'm now sketching the attention mechanics for the k=1 pooling layer: each head projects the tokens into queries and keys, and the softmax over positions gives a weighted pool, which is what I need to design the different pooling views.

I'm setting the CLS query to a fixed vector with zero weights, so every head reads the same query for the CLS token. Then I'll shape each head's keys so that the query dotted with a projected token gives a scaled direction score, letting each head focus on a chosen direction across positions.

I'm setting up the attention weights directly: a fixed query with zero projection for the CLS token, and key rows scaled by β along a chosen direction, so each head scores tokens by that direction. The uniform heads get a zero query, and the output projection is adjusted so the constant CLS residual cancels out.

I'm checking how the centering works out in the attention output: the dataset mean gets subtracted so the layer norm produces the same cosine-normalized features I used offline. Each head's block can also be scaled by its own norm. Since the k=1 features come entirely from the attention pattern and the value and output projections, the next question is which patterns to use before fitting the kernel head.

I'm looking at how to get more token-level data for evaluating pool designs, since the 64-sentence token dump is too small and reading 24 batches would be too costly. Instead I'm considering packing twelve different pool patterns into the twelve heads of a single pooling layer, each head's 64 dimensions carrying one candidate view, so one experiment can compare several designs at once.

I'm planning to fill all twelve heads with new pooling views, each carrying 64 dimensions, and then combine them offline with the full 768-dim mean I already have from M0. The open question is which 64 directions to use for each pool: either the top PCA directions of the token-level hidden states, or the most discriminative directions from LDA or ridge on M0.

I'm weighing which 64-dimensional directions to feed each pool, whether the top PCA components of the sentence-level means or a discriminative mix with ridge scores, and I'm also thinking about position-aware pools for the opening tokens of a sentence.

I'm weighing how to bias the attention toward early positions: a direction built from the position embeddings, fit on the token data, so the keys score the first few tokens of each sentence higher.

I'm setting up the first experiment: attention heads 0 through 7 each pool token evidence for one class, with the sharpness of each softmax tuned so it stays peaked without collapsing to a single token.

I'm settling the value projections for the position heads: either the top token-level PCA directions from the 1901 tokens, or the sentence-level M0 space. Scale β per head so the score spread across tokens lands near 2, with the first 2, 4, and 8 positions emphasized.

I'm considering using token-level PCA with 64 components from the 1901 tokens, since that's enough data to be stable and it stays unsupervised. The concern is that the ridge directions for the class pools are fit on all practice labels, so the attention patterns carry label information and the cross-validated kernel scores could come out a bit optimistic. The effect should be mild with 1496 samples, but I'd want a split to check it.

I'm weighing whether to keep the first eight heads as class pools and the rest as position pools, since that leaves no room for the full mean. The plan is to compress the mean into six heads and use the other six for pools, then run the first experiment (E1) to see which pools actually help before settling the final allocation.

I'm weighing whether k=2 could do better, since layer 0 would stay intact there and pooling would run over already-contextualized tokens. Given the 45-pass budget, I'll start with the cheaper k=1 experiment (E1), which needs two passes, and then reserve passes for reading the final features and verifying predictions. Before setting up position directions and the softmax temperature, I want to load the token data to get the positions.



Multi-seed results show the original's deep features reach 98.2% at 10-fold and 98.4% at 20-fold, while a 1-layer mean only gets about 97.7%. I'm next trying extra attention pools on the token embeddings to lift the 1-layer model.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\ntoks=json.load(open('pass_4/tokens.json')); mask=np.load('pass_4/attention_mask.npy').astype(bool)\nprint(type(toks), len(toks), toks[0][:12] if isinstance(toks,list) else list(toks.keys())[:3])\nX0=np.load('pass_4/token_hidden_L0.npy')  # [64,54,768]\npos=np.tile(np.arange(X0.shape[1]),(64,1))[mask]; Xt=X0[mask]\nlens=mask.sum(1); print('lens',lens.min(),lens.max())\nd=load_base(); y=labels(); M0=d['mean_hidden_L0']\nW,b=ridge_fit(M0,y,3.0)   # class directions on mean features (all 768 dims incl reserved -> L0 has no reserved issue)\nS=Xt@W  # token class scores\nprint('token score std overall', S.std(0).round(2))\n# within-sentence std\nrows=np.repeat(np.arange(64),lens)\nws=np.mean([S[rows==i].std(0) for i in range(64)],0); print('within-sent std',ws.round(2))\nnp.savez('pool_design.npz',W=W,b=b)"
}
```


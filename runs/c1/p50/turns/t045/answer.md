## reasoning summary

The 97.7% from the linear model on U is suspicious, since U includes the class-evidence dimensions built from w_c, which were fit on the full practice set. That means cross-validation on U would be optimistically biased. The kernel runs at low lambda also look off, probably a numerical issue with an ill-conditioned K, given that Y is centered but there's no intercept.

Weighing the bag-of-embeddings numbers against the test target, I'm now asking whether a contextual two-layer model could do better, since the original FFN width may matter for keeping useful features.

Weighing the parameter budget now: the fixed floor is about 24.4M, so a single layer with a 1-unit FFN lands near 26.8M, and two layers near 29.2M. Keeping the original 3072-unit FFN would push the total to roughly 38.6M, well over that. I'm also checking whether one layer can do any token-level nonlinearity at all, since the FFN only ever sees the pooled CLS vector.

I'm now weighing whether a bag-of-embeddings kernel can honestly clear the target, since it sits just under 97.8% and any CV on the class-evidence dimensions is leaky. That pushes me to consider the token-level nonlinearity that attention pooling could add, though honest validation would require per-token features I don't have from the CLS output alone.

I'm checking whether the honest evaluation can drop the class-direction value dims, though the attention weights would still carry some leakage from the fitted w_c.

I'm weighing whether a two-layer model with the original layer 0 kept intact and a pooling layer on top could capture contextual features that the bag-of-embeddings approach misses. The key tradeoff is that keeping the full 3072-unit FFN would blow the parameter budget, so I'd have to decide whether those features are worth it.

I'm checking whether a linear model or kernel on the base mean-hidden states from layers 1 and 2 does better than the bag-of-embeddings numbers. My notes already show about 98.0% for the M0+M1 kernel, though it's not clear that combining them adds anything over M1 alone. Keeping layer 0's FFN at full width doesn't fit the budget, so the open question is whether pruning it to a few hundred units keeps enough of the contextual signal to be worth the extra parameters.

I'm weighing whether attention-based pooling in a second layer could extract useful bigram-like signals from neighboring tokens, though that would cost about 29.2M parameters. A single layer with a 1-unit FFN is cheaper, and a kernel using the CLS FFN with about 768 Nyström-style units might give comparable results.

I'm going to check the honest accuracy of the best single-layer design first, and only if that reaches about 98.1% will I settle on one layer with a 1-unit FFN at 26.8M parameters. If it doesn't, I'll look at two layers.

For the single layer, I'm considering whether the twelve heads, with 64 value dimensions each, can carry enough information to the CLS position. Some uniform-attention heads could pass along a projection of the mean token embedding, which is roughly where the linear baseline already sits.

Next I'm thinking about the misclassifications, which probably need context like "not covered" versus "covered", and I'm sorting out which validation estimate I can actually trust before choosing a design.

I'm considering whether attention weights built from unsupervised token statistics, like uniform heads or ones that favor rarer content tokens, could avoid label leakage and keep the fitting in the pooler and classifier. The open question is whether supervised attention on class evidence, which probably drives the extra accuracy, can be validated honestly, since its leakage through the weights is smaller than through the values.

I'm weighing a two-fold cross-fitting scheme as the honest check: fit the class-evidence weights on each half, then score the other half, accepting that the classifier trained on the leaky half will be pessimistic on the holdout.

I'm checking whether the 98.3% number holds up, since w_c was fit on all labels and that could be inflating the class-direction values. The next step is to look at how w_c was fit and whether removing the class-direction dimensions still leaves a good estimate.

Checking how the edit weights are laid out in layer 11 to see which residual dimensions the class-direction values feed into.



The 97.7% linear CV on U is inflated, since its class-evidence directions were fit on all practice labels. I'm inspecting the layer-11 weights E1 installed to see which output dims carry that label information.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\nL='bert.encoder.layer.11.'\nqb=W(L+'attention.self.query.bias'); qw=W(L+'attention.self.query.weight')\nkw=W(L+'attention.self.key.weight'); kb=W(L+'attention.self.key.bias')\nvw=W(L+'attention.self.value.weight'); vb=W(L+'attention.self.value.bias')\now=W(L+'attention.output.dense.weight'); ob=W(L+'attention.output.dense.bias')\nprint('qw norm', np.abs(qw).max(), 'qb nonzero', np.nonzero(qb)[0], qb[np.nonzero(qb)[0]])\nfor h in range(12):\n    s=slice(64*h,64*h+64)\n    print(h, 'k norm %.2f'%np.linalg.norm(kw[s]), 'kb %.2f'%np.linalg.norm(kb[s]), 'v norm %.2f'%np.linalg.norm(vw[s]), 'rank v', np.linalg.matrix_rank(vw[s]), 'o cols norm %.2f'%np.linalg.norm(ow[:,s]))\nprint('ow nonzero rows', (np.abs(ow).sum(1)>0).sum(), 'cols', (np.abs(ow).sum(0)>0).sum())\nnzr=np.where(np.abs(ow).sum(1)>0)[0]; print(nzr[:20])\nprint('ow structure: per col nonzero count', np.unique((np.abs(ow)>0).sum(0)))\nfor nm in ['attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','output.LayerNorm.weight','output.LayerNorm.bias','output.dense.bias','intermediate.dense.bias']:\n    a=W(L+nm); print(nm, a.min(), a.max())\nprint(np.abs(W(L+'output.dense.weight')[:,0]).max(), np.abs(base(L+'output.dense.weight')[:,0]).max())\n"
}
```


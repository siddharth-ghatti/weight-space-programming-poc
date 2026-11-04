## reasoning summary

Linear looks weak here, probably because the cn*30 scaling is poorly regularized and the lambda range is too narrow. Still, the kernel results are useful: M0 alone reaches about 97.5% and M9 about 98.1%, with M0+M9 at 98.5%. M0+M1 gives 97.9%, so the gap between bag-of-embeddings and the full 9-layer stack looks like only about 0.6 to 0.9 points.

Trying a smaller two-layer design now, with a pool layer that gathers several earlier layers' features. Parameter counts look far lower than the 10-layer stack, which makes that direction worth pursuing if the accuracy holds. The concern is that the 1-layer ceiling around 97.7% sits below the target of roughly 98.3%, so I'm looking at which features could push past bag-of-embeddings.

I'm weighing whether pooling over token states with position-weighted or attention-based heads could beat the bag-of-embeddings ceiling, since E1's class-evidence and early-position pools added nothing over M0. I'm also considering whether a nonlinearity cascade (a GELU FFN on the pool layer's CLS state feeding the 768-unit tanh pooler) gives the kernel more capacity at low depth, with M0+M1+M2 around 98.0 still short of the 98.3 target.

Still weighing whether attention-based pooling could sharpen the features, though E1's class-evidence pools already came up empty. Checking how the layer counts trade off: M0 with M8 reaches about 98.3, M0 with M9 about 98.5, so the accuracy jump seems to land around layers 8 and 9.

Testing whether a kernel on the final layer's features plus a linear ridge on the early layer's scores can recover the accuracy lost by pruning.

Weighing three candidate architectures against the parameter budget, the biggest unknown is how much accuracy the pruned layers cost. I want to check the pruned M8 before committing to the 9-layer option.

I'm weighing whether the pruned layers cost much accuracy: pruned M8 and M9 land about 0.2 and 0.6 points below their unpruned versions, though the numbers are noisy. I'd like to refit with far more data than the roughly 2,500 tokens I have now, and I'm considering pulling token states for several layers in one pass instead of one pass per layer.

I'm weighing whether to skip the sequential refit and instead fit each layer's pruning on the base model's activations across many more sentences, which would give roughly 10k tokens instead of 2.5k. The trade-off is that drift from upstream layers isn't corrected, since the target is just the full FFN output on whatever input it gets.

I'm now considering a drift-correcting refit: instead of fitting the pruned FFN to the full layer output on drifted inputs, I'd set its target so that the residual sum, after layer norm, lands on the base model's next-layer hidden state.

I'm now thinking about how to pick which neurons to keep when pruning, since selecting by activation variance and FFN weight norm ignores correlations between units. A greedy forward selection that accounts for correlations, or a pivoted-QR style choice on the weighted activations, could pick a better subset. I'd also want the refit to use paired drifted and base token states from the same sentences.

I'm working out a greedy forward selection over the 3072 units that picks the subset which best explains the full FFN output. Doing it naively looks too expensive, so I'm considering residualizing the Gram matrix at each step, which might make the scoring tractable.

I'm weighing whether the numbers and token budget justify a larger refit, since fitting 3072 units on about 2,500 tokens leaves the Gram matrix underdetermined. Greedy selection with residualized Gram updates looks affordable at this size, but it only pays off if I get substantially more token states first. Before choosing an approach, I need to know how many passes the remaining budget allows.

Now I'm weighing the data budget: four passes would give 256 sentences with every layer's states, around 10k tokens, which is far more than the current fit has. The catch is that the FFN input is the post-attention layer-norm state, which the layer-boundary reads don't expose directly, so I'd need the earlier zero-FFN trick to get at it.

I'm checking whether computing the FFN activations in numpy from the read hidden states counts as reimplementing the forward pass, and I'm settling on keeping that step limited to closed-form fitting on those activations.

I'm weighing whether the FFN input can be exposed without editing: by zeroing layer l's FFN and truncating the stack to l+1 layers, its normalized attention output becomes the final hidden state. That still means one pass per layer, so getting 256 sentences per layer takes four passes each.

I'm considering reading the attention probabilities at layer l and recomputing the attention sublayer's context from the hidden states, so the FFN input could come out without zeroing any FFN.

I'm weighing whether the 3072-unit FFN can be fit with far more token states than the 2,500 I have now. One option is exposing each layer's post-attention normalized input, but that takes a separate pass per layer, so the cost adds up quickly. Before going further with that, I want to see how much accuracy the small-k combinations can reach.

I'm going to check combinations of the first few layers, like M0 with M2 or M0 through M3, and see whether adding CLS features helps. First-token pooling might also carry information, since sentence openings like definitions or exclusions can be telling. The token-level data is too small to test those variants on the full set. I'm also thinking about n-gram-like features, since layer 0's attention already mixes in neighboring tokens, which may be why M1 adds to M0. I'll quickly test these combinations on the base features to see how far small-k models can go.



Bag-of-embeddings alone reaches 97.5%, and the large gain only appears at M8–M9, so I'm now testing multi-layer combinations of the early layers to see how small k can go before accuracy drops.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz')\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef K(Xs,g=1.0):\n    Z=np.concatenate([cn(X) for X in Xs],1)/np.sqrt(len(Xs)); return np.exp(g*(Z@Z.T-1))\nM=lambda l: d[f'mean_hidden_L{l}']; C=lambda l: d[f'cls_hidden_L{l}']\ncombos={'0,1,2':[M(0),M(1),M(2)],'0,1,2,3':[M(0),M(1),M(2),M(3)],'0..4':[M(i) for i in range(5)],'0..6':[M(i) for i in range(7)],\n        '0,2':[M(0),M(2)],'0,1,C1':[M(0),M(1),C(1)],'0,C1':[M(0),C(1)], '0,1,9':[M(0),M(1),M(9)],'0,4,9':[M(0),M(4),M(9)],'0,8,9':[M(0),M(8),M(9)],'1,9':[M(1),M(9)]}\nfor nm,Xs in combos.items():\n    for g in [1.0,2.0]:\n        print(nm,g,krr_cv(K(Xs,g),y,[0.003,0.01],k=10,seeds=(0,)))"
}
```


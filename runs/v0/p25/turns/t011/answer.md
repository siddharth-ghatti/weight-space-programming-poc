## reasoning summary

I'm estimating the ceiling at about 98.3% using mean0, mean11, and cls11 together, versus 97% for mean11 alone, so roughly 20 more correct out of 1496 — that seems worth pursuing given the budget. Now I'm wondering how to get the bag-of-embeddings signal from layer 0 into the final layer, maybe by routing it through the CLS token.

I'm weighing whether to route the layer-0 bag-of-embeddings signal through the CLS token, which would let it persist to the final layer. The catch is that CLS is transformed by layers 1 through 10, and changing its state shifts the keys that other tokens attend to, so every token's representation could be perturbed. I'm looking for a way to carry that signal without disturbing the rest of the network.

I'm weighing whether layer 10 could also pool a mean into the CLS token, since mean10 isn't otherwise available at layer 11, but the modified CLS state would still shift the keys other tokens attend to in later layers.

I'm considering whether a head in layer 10 could attend uniformly only from the CLS query position, using the fact that CLS's hidden state looks distinct from other tokens. The hope is to build a query that responds to CLS-specific features while leaving the other tokens' layer-10 outputs unchanged. I can't tell yet whether such a direction exists or how well it would separate CLS from the rest.

I'm considering whether heads in layer 11 could pool the full 768-dimensional layer-11 input in a more useful way than uniform averaging alone, since all twelve heads are already in use. Any extra head would have to replace an existing one, which would perturb other tokens' representations somewhat.

I'm considering using layer 11's twelve heads for attention-based keyword pooling instead of uniform averaging: a head whose key points along a class-discriminant direction would focus on the tokens most indicative of each class, a nonlinear feature that doesn't depend on sentence length. Measuring whether this beats the plain mean would require token-level hidden states.

I'm wondering whether the layer-11 heads could do something beyond uniform averaging, like attention-based pooling that picks out the tokens most indicative of each class. Checking that would need token-level states, which are costly to collect across all 1496 texts, so I'd have to evaluate it heuristically with full passes.

Now I'm weighing a hybrid where layer 0 pools mean0 into the CLS token, with later layers left to carry it forward while the rest of the network is simplified.

I'm checking whether the CLS token can carry mean0 forward, but LayerNorm rescales it each layer and the attention and FFN updates are about as large as the residual, so a big stored magnitude wouldn't stay dominant. Shared LayerNorm gains also limit how far CLS can be singled out, and I'm not sure what that does to the other tokens yet.

Next I'm checking whether a cheaper way exists to get more views in layer 11, using different heads that attend over the layer-11 inputs in different ways rather than all uniformly. Honestly, a 97% versus 98% gap may not matter much, but there's plenty of budget, so I'll first build the basic version and verify it, then explore improvements.

I'm building a basic version that combines cls11 with a scaled mean11, and I'll run a quick cross-validation over a few scaling factors on the raw features before the layer norm changes things.



Combining views reaches about 98.3% versus 97% for mean11 alone, leaving roughly 1% headroom. I'll build the simple pooled-CLS version first and verify it end to end, then test linear mixes of the original CLS with mean11 to see how much CLS to keep.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nprint('norms cls11',np.linalg.norm(c11,axis=1).mean(),'mean11',np.linalg.norm(m11,axis=1).mean())\nprint('std across samples: cls11',c11.std(0).mean(),'mean11',m11.std(0).mean())\nfor c in [0.5,1,2,4,8,1000]:\n    X=c11+c*m11\n    r=[cv(X,y,folds,lda_fit,a)[0] for a in [0.1,0.2,0.3]]\n    r2=[cv(X,y,folds,ridge_fit,l)[0] for l in [30,100]]\n    print(c,' '.join(f'{v:.3f}' for v in r+r2),flush=True)"
}
```


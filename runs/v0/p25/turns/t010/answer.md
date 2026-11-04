## reasoning summary

I'm settling on the errors being mostly truly ambiguous, so around 97% looks like a practical ceiling, with 98% maybe reachable. I'll build the basic pipeline first and only then look at improvements with what's left of the budget. Now I'm working out the edit for layer 11, setting the query weights and bias to zero so attention goes uniform, and considering whether the value projection should be the identity.

Still setting the layer 11 weights: the value projection is the identity, and the output projection is scaled identity. I'm zeroing the feed-forward block and the LayerNorm offsets so the CLS token passes through with only the attention mixture added. The pooler will be a scaled identity, and I'm noting that the outlier dimensions will stay near-constant across samples.

I'm still weighing whether the pooler and LayerNorm will distort the linear classifier. Since LN divides by the norm of the projected vector, the logits get a per-sample scale, which could matter if that norm varies a lot across inputs.

I'm considering whether I can evaluate different mixing weights for the CLS and mean vectors offline, since with the edit the layer-11 output is just their combination before the LayerNorm. Computing that in numpy from activations I already have would let me sweep the configurations without extra forward passes, though I'll need to confirm the mean is taken over the same tokens.

I'm weighing whether to simulate the LayerNorm on the activations I already have, but I'll skip emulating it and stick to linear-combination analysis on the features. Once that's done, I'll confirm the choice with real forward passes.

I'm weighing whether a designed mixing of the CLS and mean vectors can reproduce the concatenated classifier's logits exactly, though refitting on the combined vector might not recover them. The LayerNorm and tanh in the pooler could still distort things, so I'm considering applying the classifier directly and checking the effect.

I'm weighing whether the mixing matrix M can be chosen so that the combined vector reproduces the concatenated classifier's discriminative directions, even though 1536 dimensions can't fit into 768 without loss. Since the useful subspace is small, I'm considering whether a designed M can keep the important part of that information.

Checking whether a mixture of the CLS and mean vectors, or some designed combination of them, could recover the concatenated classifier's accuracy. A combined vector would give roughly 0.973–0.975 CV, a gain of about half a percent over the mean alone, which is small but possibly real given the sample size. I'm also wondering whether anything with a bigger payoff exists, since mean0 plus mean11 reaches about 0.978.

I'm checking whether combining several layers' mean vectors might push accuracy toward 0.98, which would be worth the extra routing. I'll test a concatenation of all 13 means with ridge to find the ceiling first.



Accuracy is probably near its ceiling at about 97%, since most errors are truly ambiguous sentences. I'm checking whether pooling features from several layers would improve it enough to justify a more complex routing.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\ndef show(name,X,alphas=[0.1,0.3],lams=[30,100,300]):\n    X=X.astype(np.float64)\n    r=[cv(X,y,folds,lda_fit,a)[0] for a in alphas]\n    r2=[cv(X,y,folds,ridge_fit,l)[0] for l in lams]\n    print(name,'LDA',' '.join(f'{v:.3f}' for v in r),'| ridge',' '.join(f'{v:.3f}' for v in r2),flush=True)\nshow('mean11 ridge fine',D['mean11'],[0.15],[30,60,100,200])\nshow('mean0+4+8+11',np.hstack([D[f'mean{L}'] for L in [0,4,8,11]]),lams=[100,300,1000])\nshow('all means',np.hstack([D[f'mean{L}'] for L in range(13)]),lams=[300,1000,3000])\nshow('mean0+mean11+cls11',np.hstack([D['mean0'],D['mean11'],D['cls11']]),lams=[100,300,1000])"
}
```


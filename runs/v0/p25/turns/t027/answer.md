## reasoning summary

I'm noticing the mean-token representations shift by roughly half their across-sample spread from layer 2 on, which is a big change. My guess is that it's because the CLS token changed and other tokens attend to it in early layers, since CLS acts as an attention sink in some heads. Still, the mean features hold up, reaching 0.971 at layer 11 versus 0.967 for the base model, and combining cls11' with mean11' gives 0.975.

Layer 0 pooling doesn't help as written: the CLS token's copy of the mean information gets diluted by later processing, reaching only 0.936 at layer 1 against 0.961 for the true mean. Keeping the mean-based features at layer 11 gives a marginal gain (0.971 against 0.967), but adding the CLS features only reaches 0.975 with no further improvement, so the edit isn't worth the risk and I'll revert it. I also want to understand why the bag-of-words information fails to carry over into CLS at layer 11 when the combined model reached 0.983 earlier.

I'm weighing whether to pool across all layers by having CLS carry a running mixture, though it's unclear it would hold them all. A different direction looks more promising: the low-layer means add under a percent each to the [cls11, mean11] pair, but layer-11 tokens can't expose that information unless it's written into them. That points toward reserving dimensions across layers 1–10 to carry a low-layer signal that the final uniform pooling can recover, which would need careful control of reads and writes on those dimensions and of the LayerNorm gains.

I'm considering whether the reserved dimensions could hold the original embedding coordinates, since if layers 0 through 10 leave those dimensions untouched, the LayerNorms would only rescale and shift them, so the signal survives to layer 11. The catch is that BERT uses those same coordinates for its own computation, so reserving them would cost something I'd need to check.

I'm weighing whether to reserve a subset of coordinates, roughly 32 of them, to carry the original embedding values through the layers, which would cost about 4% of BERT's residual stream. The open question is which coordinates to pick: greedily choosing them by univariate statistics on the labels risks leakage, though it may be mild.

I'm now checking whether reserving a subset of coordinates for the original embedding values would work, since layer-norm drift could be controlled by zeroing writes to those dimensions. The concern is that BERT itself uses those coordinates, so I'd need to measure the cost, and I'd want to pick the subset with nested cross-validation to avoid leakage.

Still weighing whether to keep the mean11 quality for the reserved-dims idea, since it's about 0.5% in potential gain against the risk of disturbing BERT's own coordinates. I could also try a cheaper route: pool at layer 11 with attention that skips [CLS], [SEP], and punctuation, which carry large norms but little information.

I'm considering pooling at layer 11 with several attention heads that behave differently. One could weight tokens by their projection onto a class-evidence direction, so a strong keyword anywhere stands out instead of being averaged away. That would be a nonlinear feature, though its gain is uncertain.

I'm now considering max-like pooling, where each class has a head that weights tokens by their evidence score so a strong keyword isn't washed out by averaging. The evidence per token would be the LDA direction applied to each hidden state, which matches the mean since the mean is linear in those scores. Testing it would take about two passes per trial.

I'm considering a max-like pooling variant for layer 11, where a few heads weight tokens by their class-evidence scores so a strong keyword isn't diluted by averaging. The remaining heads would keep a uniform mean over a reduced projection, and I'd check whether an LDA-friendly PCA subspace of the mean loses anything.

I'm going to make the output projection a scaled identity so the head outputs dominate the residual, leaving cls11 as minor noise. That skips the designed superposition, and the refit classifier on the pooled features should absorb the rest.

I'm weighing max-style pooling heads against reserving dimensions for low-layer signal. Without token-level states I can't estimate the first offline, so the reserved-dimensions idea is the one to test, starting with the existing mean0 data to see how much signal it could carry.



The layer-0 [CLS] trick failed: bag-of-words info is mostly gone by layer 11, and it disturbed other tokens, so I'm reverting it and will estimate the reserved-dimensions idea offline next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\ndrop_edit('bert.encoder.layer.0.attention.self.query.weight')\nprint(edits())\nwith open('notes.md','a') as f: f.write(\"\"\"- run22: L0 CLS-uniform trick RESULT: cls1'=0.936, cls11'=0.949, cls11'+mean11'=0.975 (no gain), other tokens' means shifted ~50%. REVERTED (dropped L0 edit).\n\"\"\")\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nm0=D['mean0'].astype(np.float64); c11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nrng=np.random.default_rng(0)\nbase_X=np.hstack([c11,m11])\nfor k in [16,32,64]:\n    for trial in range(2):\n        R=rng.choice(768,k,replace=False)\n        X=np.hstack([m0[:,R],c11,m11])\n        print('random',k,round(cv(X,y,folds,ridge_fit,300)[0],4),flush=True)\n# F-stat chosen (leaky, just for potential)\nmu=m0.mean(0); F=np.array([np.var([m0[y==c,d].mean() for c in range(8)])/m0[:,d].var() for d in range(768)])\nfor k in [16,32,64]:\n    R=np.argsort(-F)[:k]; X=np.hstack([m0[:,R],c11,m11]); print('Fstat',k,round(cv(X,y,folds,ridge_fit,300)[0],4),flush=True)"
}
```


## reasoning summary

I'm wondering whether to test the model on some hand-written sentences that fall outside the test distribution, using read or predict. Predict seems to only accept practice ids, while read can take raw text and return logits, so I could run a sanity check that way later. Next I'm turning to which multi-layer feature improvement is most feasible, starting with an idea about pooling.

I'm weighing a multi-layer feature improvement now, and one idea is to pool the layer-11 hidden states while folding in a constant per-sentence vector from the earlier layers. That's appealing, but the CLS token at layer 11 is built entirely from layers 0 through 10, so I can't inject a bag-of-embeddings signal there without changing those earlier layers. I'm checking whether the CLS input at layer 0 is constant across sentences, since that might let me shape its attention without touching the other tokens.

I'm checking whether a rank-one tweak to the layer-0 query weights could make the CLS token attend uniformly, but any change to the other tokens' queries would also shift their attention, and I'm still trying to find a direction that leaves them alone.

I'm weighing whether a rank-one change to the layer-0 query weights could shift the CLS token's attention to uniform without disturbing the other tokens. The catch is that the other tokens' embeddings aren't exactly orthogonal to the CLS direction, so any correction would only hold approximately. Reading the actual layer-0 token vectors would let me fit that correction more carefully, though rebuilding the embedding layer by hand feels fiddly.

I'm now thinking that the CLS token at layer 0 could attend uniformly in each head, so its output would carry the mean of the layer-0 token vectors, spread across all heads to fill the full width. The CLS state would then pass through layers 1 to 10 as usual, but since other tokens attend to it in the middle layers, that change would also reach them.

The simpler options seem to be reweighting the layer-11 mean or pooling over a subset of tokens, though those look minor. I could also take layer 10 and layer 11 together.

I'm considering whether pooling at layer 10 could work instead: layer 10 would then average its own inputs, but every token would receive the same attention output, so the tokens would collapse toward one vector unless the mixing weight stays small. With a small weight the per-token information would survive, and the layer-11 pooling would still have something to average.

I'm considering whether a layer-11 pooling scheme could split its heads, with some doing uniform mean pooling and others doing attention-weighted pooling for token importance. That would need token-level features, which makes it too involved. Given that the current setup already reaches about 98% in cross-validation, the remaining gain from multi-layer features looks small and probably within the noise of the test set.

I'm still weighing whether pooling at layer 9 could work, since layers 10 and 11 would both read the modified CLS token, and the main worry is that other tokens attend to it too.

I'm now looking at whether pooling at layer 10 would help: CLS's attention to other tokens at that layer would change the mean at layer 11 for everyone. Modifying CLS only at the output of layer 10 would avoid that, since layer 11 pooling would still include CLS's own output with equal weight.

I'm checking whether the CLS token at layer 10 can be pooled on its own, which would require a query change that affects only CLS. That depends on how distinct h10 for CLS is from the other tokens, so I want to measure that using per-token layer-10 vectors.

I'm now trying a key-side trick instead: adding a constant component along one direction to every key, so that CLS's attention can be shaped without changing how the other tokens attend.

Checking whether a key-side shift could shape CLS's attention, but a constant offset cancels out in the softmax, so it can't flatten the distribution. The only lever left is the query's scale, which doesn't work per-token either. I'm now considering whether CLS's output could go through the value path instead.

I'm considering repurposing one of layer 10's heads to attend uniformly and carry a projection of the layer-10 states, scaled small so other tokens barely change. The worry is that removing that head's original output would perturb the tokens more than the added signal helps.

I'm weighing whether removing one of layer 10's heads would matter much, since head pruning often costs little. The worry is that the added signal would be small relative to the variance already in the layer-11 mean, though a standardized, ridge-regularized kernel might still pick it up.

I'm comparing a few variants by cross-validation: the current single-scale kernel features against a multi-scale version that combines two bandwidths, which still fits within the feature budget. I'll also check whether standardizing or whitening the features works better.

I'm going to try PCA-whitening the features as well, and then just run the comparison.



The full build checks out at 100% on the 512 practice items, so the checkpoint is safe. Next I'll try cheap refinements averaged over three fold splits, and I'm deferring multi-layer pooling as too risky for about 0.5%.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy'); E=np.load('feats/edA_cls12.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef mkfolds(seed):\n    rng=np.random.default_rng(seed); fo=np.zeros(len(lab),int)\n    for c in range(8):\n        idx=np.where(lab==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%5\n    return fo\nFO=[np.load('feats/folds.npy'),mkfolds(7),mkfolds(8)]\nR=np.sqrt(768)\ndef run(scales,lams,rho=1.0,ts=None):\n    ts=ts or [0]*len(scales)\n    res={l:[] for l in lams}\n    for fo in FO:\n        FS=[]\n        for f in range(5):\n            tr=fo!=f; mu=E[tr].mean(0); sd=E[tr].std(0)+1e-6\n            Z=(E-mu)/sd; C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True); U=(Z@C.T)/R\n            FS.append(np.concatenate([gelu(a*U+t) for a,t in zip(scales,ts)]+[rho*Z/R],1))\n        for lam in lams:\n            pr=np.zeros(len(lab),int)\n            for f in range(5):\n                tr=fo!=f; te=~tr; F=FS[f]; Ftr=F[tr]; m=Ftr.mean(0); Y=np.eye(8)[lab[tr]]-1/8\n                Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n                pr[te]=((F[te]-m)@Wt).argmax(1)\n            res[lam].append((pr==lab).mean())\n    return {l:round(np.mean(v),4) for l,v in res.items()}\nprint('a=2', run([2],[0.3,0.5,1]))\nprint('a=1.5', run([1.5],[0.3,0.5,1]))\nprint('a=1,3', run([1,3],[0.3,1,3]))\nprint('a=2,t=-.5', run([2],[0.3,1],ts=[-0.5]))\n"
}
```


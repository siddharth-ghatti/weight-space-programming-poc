## reasoning summary

The refit only brings the error down a little, from 0.407 to 0.344 at m=1024, and pruning still drops about 34% of the FFN output norm there, so the FFN looks spread across many units rather than concentrated in a few. Since GELU blocks exact merging of units, I'm considering also adjusting the kept units' input weights (W1 rows and b1), not just their output weights.

I'm weighing whether unit merging or clustering the kept units in their input-weight space could recover what pruning loses, but summing output weights of clustered units doesn't seem to capture it well. Greedy forward selection might beat the energy-based pick, though the gap from 3072 to 1024 units is still large. Pass_19 (energy plus refit) clearly beats the std-selected pass, so the selection and refit matter more than the modest offline error suggests.

I'm weighing a drift-correction angle, where each layer's pruned FFN partly compensates upstream error, though only the part predictable from the kept units' activations can be recovered. I'm also considering whether combining the pruned M9 features with the base M0 features would recover some of the discriminative loss, since the classifier only needs the features to stay discriminative rather than match exactly.

I'm checking whether the pruned network's combined features, like M0 with pruned M9, could match base M9 accuracy, since that would be good enough. Carrying several layers' information forward would need reserved dimensions and carry heads, and I'm weighing how much capacity that gives.

Weighing the reserved-dimension cost against layer count: each extra reserved dim takes capacity from content processing. Comparing parameter budgets, a 1536-unit FFN with 9 content layers plus pool comes to about 71.7M parameters, roughly a 34.5% cut, while 1024 units gives 63.8M and 2048 gives 79.6M. Dropping one middle layer and keeping m=1536 would be another option, leaving 9 layers in total.

I'm looking at whether the pool layer's feed-forward block can do useful nonlinear work on the pooled CLS vector, or whether it would just sit as wasted parameters. I'm also noting that the final layer only ever serves the CLS token, so it could double as the pooling layer, with the features taken as the mean of the layer-8 output.

I'm weighing whether the base KRR features from M9 and M8 leave enough accuracy margin, since the threshold is tight at around 97.8 against a base near 98.1 to 98.5. I'm also wondering whether the pooler layer's feed-forward block could add kernel-like capacity on the pooled vector, though the network approximation may give up some of that.

I'm weighing whether dropping a couple of whole layers might cost less than pruning FFN units across all of them. The earlier pass that dropped two layers outright landed close to the m=1024 pruning result, while saving fewer parameters, so pruning looks like the more efficient route.

I'm thinking about ways to make the pruning less lossy: correcting the drift against base hidden states, picking units by greedy forward selection instead of energy, and possibly refitting the LayerNorm parameters or attention output too. The attention refit is harder, since it needs the context vectors, which I don't have directly. I'll start by checking greedy selection against energy at m=1024 on layer 4 using held-out error.

I'm thinking about getting the column subset selection right, since the Gram matrix over the training tokens is rank-deficient and the ridge term has to carry the regularization. With only 1717 training tokens against 1024 units, the fit will overfit unless I lean on more data, and the 64-sentence refit set could be widened to the full practice set.

I'm weighing a larger hidden width of 1536 units, which brings the offline error down to about 0.25 instead of 0.34 at 1024, at the cost of roughly 8M more parameters. Since failing the threshold is far worse than a slightly larger model, I'm leaning toward safety first. Next I want to check whether pruned M9 combined with base M0 holds up against base M0 with base M9 at 1024 units.



Refit only recovers a little: at m=1024, held-out FFN output still drops about 34%, and the residual stream about 15%, since the computation is spread across many units. Next I'll test whether combining M0 with pruned M9 (possibly multi-layer) recovers base quality on the 1024 subset, since M0 is cheap to carry.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz'); P=practice(); id2i={p['id']:i for i,p in enumerate(P)}\nids19=json.load(open('pass_19/meta.json'))['ids']; idx=np.array([id2i[i] for i in ids19]); yy=y[idx]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef K(Xs,g=1.0):\n    Z=np.concatenate([cn(X) for X in Xs],1)/np.sqrt(len(Xs)); return np.exp(g*(Z@Z.T-1))\nE=lambda l: np.load(f'pass_19/mean_hidden_L{l}.npy'); B=lambda l: d[f'mean_hidden_L{l}'][idx]\nfor nm,(xe,xb) in {'9':([E(9)],[B(9)]),'0+9':([E(0),E(9)],[B(0),B(9)]),'0+8':([E(0),E(8)],[B(0),B(8)]),'0+5+9':([E(0),E(5),E(9)],[B(0),B(5),B(9)]),'0+8+9':([E(0),E(8),E(9)],[B(0),B(8),B(9)])}.items():\n    print(nm,'edit',krr_cv(K(xe),yy,[0.003,0.01],k=10,seeds=(0,1)),'base',krr_cv(K(xb),yy,[0.003,0.01],k=10,seeds=(0,1)))"
}
```


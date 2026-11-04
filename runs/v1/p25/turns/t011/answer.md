## reasoning summary

I'm checking whether the layer-0 mean feature, the bag-of-embeddings-plus-position average, gets 97.5% when concatenated with M11 in standardized space, since the low-variance directions turn out to matter. Low-dimensional PCA add-ons didn't help. Now I need to work out how to actually compute M0.

I'm weighing where the M0 signal could enter the final CLS. Pooling at layer 0 would get mixed by the later layers, so I'm looking at pooling in layer 11 instead, where token identity may still be roughly linearly readable, though the signal may sit in low-variance directions. A nonlinear per-token transform before pooling, perhaps in layer 10's FFN, is another option I'm considering.

I'm now weighing a repurposing idea: since only the CLS output of layer 11 matters, layers 10 and 11 could be retooled as a readout network, pooling h10 instead, with the constraint that h11 changes are harmless. The open question is how to get both M0 and M11 (or M6) into the final CLS, given that the concatenation results so far favor [0, 6, 11] at 97.9%.

I'm now checking whether a direct linear readout of token identity from the layer-11 token states could recover the layer-0 mean, which would make the ridge result on M11 surprising. Since the ridge on the standardized concatenation still gains from M0, I suspect the token identity is not linearly recoverable from the mean-pooled M11 alone.

I'm looking at whether the layer-0 mean could reach the final CLS through the CLS token itself, since its layer-0 state is the same for every sentence and changing it would alter how later layers attend to it. I'm also asking whether pooling a broadcast of M0 into a low-energy subspace for all tokens would avoid that.

I'm checking whether class-directed LDA on M0, keeping about 7 dimensions fitted in-fold, combined with the full M11 features, can reach the same accuracy, since that would mean transporting only a handful of dimensions from layer 0. I'm also considering a simpler stacking approach, summing separately fitted ridge scores from M0 and M11.

I'm testing whether an ensemble of separately fitted ridge models on M0 and M11, summing their scores, comes close to the combined concatenation. If it does, only about 8 scores from layer 0 would need to reach the final CLS. I'm considering writing those scores into the layer-0 CLS, which is constant across sentences in base BERT, though any perturbation there would reach the other tokens through attention.

I'm working through the idea of carrying the layer-0 scores to the final CLS through its own residual stream, but in layer 11 the CLS output is mostly the pooled mean, so its own state would be lost unless the final layer keeps a path for it. I'm considering letting the layer-11 attention add a moderate mean term while leaving some residual dimensions of the CLS state intact to hold the score signal.

I'm wondering whether layer 11's attention can pool any linear function of the token states, and whether the residual stream keeps some of the layer-0 embedding information. I'm considering adding a sentence-level constant to every token at layer 0, though that might disrupt the model unless the magnitude stays small and the directions are low-energy. I also want to check whether BERT's rarely used hidden dimensions, such as the large-magnitude outlier dims, could carry the signal.

I'm considering whether BERT's low-variance hidden directions could carry a signal, though the downstream weights would still read them at full strength. I'm now thinking about whether tokens attend much to the CLS token, and I want to measure how much attention flows there across a sample of texts.

I'm now wondering how much attention non-CLS tokens actually send to the CLS position, layer by layer and head by head, since that bounds how much changing CLS could disturb the rest of the sentence. I'm also doubting whether the CLS residual keeps early information across the layers, and I'm considering an accumulator design in which the CLS row works differently at every layer.

I'm working out how to set the CLS state so its query nearly vanishes in one head, by writing a large marker into it so the attention it sends is uniform, while leaving the other tokens' queries unchanged.

I'm working out whether the CLS query can be made nearly uniform in one head by writing a large marker into its state, since layer norm would cap the magnitude, and I'm checking whether the leftover CLS-specific term stays small enough.

I'm checking whether a per-token pooled offset could carry the signal, but the marker would dominate the CLS state and leave little room for it. I'm also starting to consider a broader alternative: pooling at an intermediate layer into every token.

I'm weighing a readout network built from the last two layers, with the CLS row handled by two attention and FFN blocks, as a way to get the extra percentage point that the single-layer M11 features miss.

I'm weighing whether token identity can be read out of the layer-11 token states. The MLM head decodes identity from the final layer, so that path might be less lossy than I assumed. I'm also considering whether nonlinear per-token features from layer 10 could pool into something like the layer-0 mean.

I'm considering whether attention-based keyword pooling at layer 11 could capture discriminative token features better than the mean, using class-directed queries derived from the sentence-level LDA directions.

I'm now sorting out the trade-offs: a layer-11 uniform pool with the classifier fit on M11 is the safe baseline, expected around 96%, and it needs no extra passes to fit, only one to verify. Adding FFN random features should push it to about 96.5%.

I'm weighing a multi-layer design where the CLS state accumulates information from earlier layers, aiming for around 97.5%. I'm also considering a variant where layer 0's pooled signal is kept intact by giving the CLS state a large marker, though layer normalization limits how much that can preserve.

I'm weighing whether the residual path can keep CLS information intact through layer 11, given that layer norm and the attention and FFN outputs will shift the state. Rather than preserving the layer-0 signal exactly, I'm considering whether the final readout can instead learn from a degraded but still informative CLS state, since it's fitted on the actual perturbed features.

I'm considering having layer 11's attention add a scaled mean of the token states into the CLS residual, which already carries the layer-0 signal. The concern is that the two parts would be summed in the same 768 dimensions and couldn't be cleanly separated, unless the mean is projected into a subspace the residual leaves free. The CLS residual itself is added in full, so I can't project that part the same way.

I'm considering whether the CLS position could subtract its own projected state with a dedicated head, but that would need CLS-specific attention, which the uniform pool doesn't give me. Since the CLS state is also part of the mean, I can't cleanly separate the two terms that way. I'm now looking at whether the feed-forward block in layer 11 could do that removal instead, though it would only approximate a linear projection.

I'm checking whether the FFN can implement an arbitrary linear map, since GELU(z) minus GELU(-z) gives exactly z, so paired units yield a linear transform of rank up to 1536 in one layer.

I'm thinking about how the FFN could strip out the CLS state's own component in a subspace, since the attention output and the residual get summed together before the FFN sees them. The clean fix would be to have attention write the mean-pooled signal into a subspace S while the CLS residual keeps its information in the complement, so that the two parts don't mix.

I'm checking whether splitting the 768 dimensions into a subspace for the mean-pooled signal and a complement for the CLS residual could keep both pieces separable. The uncertainty is how much information survives projecting each part down to 384 dimensions, which I'd have to test offline.

I'm testing whether the CLS state at layer 11 in the base model, combined with the uniform layer-11 mean M11, could give a useful readout. If so, I'd write the final state as a sum of the CLS residual and a projected M11 term, keeping them in separate subspaces so they don't collide.

I'm now testing whether simply summing the CLS state with a scaled layer-11 mean, LN(C11 + c·M11), works across a range of scales c, which would be the first design with no projection at all. If it beats the 96.5% baseline at some c, that would settle it, and a concatenation of the two vectors is worth checking alongside it.

I'm going to compare the CLS state at layer 12 and the layer-11 mean, and also try a concatenation of CLS at layer 11 with M11 plus the layer-0 mean, to see which combination holds the most information.



Adding the layer-0 mean to mean(h11) lifts accuracy to about 97.5–97.9%, against 95.9% for M11 alone, while low-dimensional PCA summaries of the earlier layer don't help. Next I'll test cheap combinations of the final [CLS] state with M11, both concatenated and superimposed.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef LN(X): \n    X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\nfor name,Xs in [('C11+M11',['C11','M11']),('C12+M11',['C12','M11']),('C1+M11',['C1','M11']),('C11+M0+M11',['C11','M0','M11']),('M0+M10',['M0','M10']),('M0+M9',['M0','M9']),('M1+M11',['M1','M11'])]:\n    fn=lambda tr: np.concatenate([stdz(F[k],tr) for k in Xs],1)/np.sqrt(len(Xs))\n    print(name,[cv(fn,l) for l in [10,30,100,300]])\nfor c in [1,3,10,1000]:\n    X=LN(F['C11']+c*F['M11'])\n    print('LN(C11+c M11)',c,[cv(lambda tr: stdz(X,tr),l) for l in [10,30,100,300]])\nprint(np.linalg.norm(F['C11'],axis=1).mean(), np.linalg.norm(F['M11'],axis=1).mean(), np.linalg.norm(F['M11']-F['M11'].mean(0),axis=1).mean())"
}
```


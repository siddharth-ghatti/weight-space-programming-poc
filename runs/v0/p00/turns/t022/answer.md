## reasoning summary

I'm weighing whether bigram heads are worth it, since the residual errors are mostly semantically hard and I have no forward passes to spare. I'm also looking at lower-risk improvements, starting with the fact that the test set has roughly the same distribution as practice, where cross-validation gave 97.4%.

I'm checking the HF implementation details I can't verify directly: the attention mask handling, eval-mode dropout, and the GELU activation in the config. Those look fine on paper, so the main risk is whether the harness calls model.eval().

I'm checking the edit semantics: W(name) returns the saved edit when one exists, so saved arrays act as full replacement tensors rather than deltas, and the classifier starting at zero is consistent with that. I still need to decide whether bigram heads are worth trying, given how few forward passes I have.

I'm weighing whether a targeted approach would be simpler than full bigram heads, such as a feature that just checks whether the previous token is "not". I'm also considering a windowed attention head over neighboring positions, which would be easier to make robust but more complex overall.

I'm weighing a bigram version that would only fire when the current token and the previous token both match strongly, so a diffuse attention pattern would just leave it inactive and fall back to the unigram scores.

I'm going to check whether a bigram correction fitted to the unigram model's residuals can be added safely, so that if the bigram signal fails to fire, the model falls back to the unigram baseline. I'll test how much gain a limited set of frequent bigrams gives before deciding whether it's worth the code.

I'm working out how to get a previous-token signal into the residual stream, since the unigram neurons already act as near one-hot identity detectors that a copying head could carry forward.

I'm planning a reserved block of residual dimensions to carry a compact identity code for the first tokens of important bigrams, written by the layer 0 feed-forward neurons.

I'm sketching a layer-1 feed-forward design where each bigram neuron fires only when the current token matches its second token and the previous token's code matches its first. I'm reserving a few extra residual dimensions for that code so the previous-token signal can be carried forward.

I'm thinking about how to give the layer 1 previous-token signal a place in the residual stream, possibly by reserving a second block of dimensions that gets copied from the prior position. I'm also weighing whether position can be encoded per token in layer 0 instead of relying on attention to find the neighbor.

I'm checking whether position can be decoded linearly from the layer 0 residual, using sinusoidal features fit by least squares over random tokens and positions. The point is to see how cleanly a previous-token head could target the neighboring position.

I'm checking the kernel sum for a offset of one position: with frequencies up to 32 over a length-128 range, the sidelobes at distance one are small but not negligible.

I'm checking whether a 32-frequency Dirichlet-style kernel gives enough separation between the peak and the neighbors at distance one, and the gap is about 12 units of score. I'd need a scale factor of roughly 4 to keep most attention on the correct position, and since each head has 64 dimensions, 32 cosine/sine pairs fit exactly.

I'm considering whether BERT's learned position embeddings could support a precise previous-token head without high-frequency components, since adjacent positions look fairly similar in cosine terms. I'm also noting that BERT's own middle layers apparently handle previous-token attention through learned weights, though I can't verify that here. Given I can't check any of it with forward passes, a subtle bug is a real risk, and the expected gain is only about 1%, even if the graceful-degradation design limits the downside.

I'm considering the failure case where the previous-token head attends to the wrong position. If attention stays diffuse, the prior-token code stays low and the bigram neurons won't fire, so the model falls back to unigram scores. If it lands on a neighboring position instead, the neurons would fire on mismatched pairs, though those are often semantically close anyway. I'm also checking what happens if attention lands on the current position itself.

I'm checking whether a limited set of frequent bigrams, with at most about 3072 pairs, gives a worthwhile gain, and whether the previous-token signal can be kept isolated in its own reserved dimensions so it does not disturb the unigram pathway.

I'm weighing whether to fit bigram corrections on the unigram model's training residuals, but since the unigram table already fits the training data perfectly, those residuals are too small to be useful. The alternative is fitting unigrams and bigrams jointly, which keeps the fallback property only loosely.

I'm weighing a naive Bayes version for the bigram terms, since NB weights are independent per feature and the unigram table would stay unchanged whether bigrams are present or not. I'm going to test a blend of the unigram scores with restricted bigram log-ratios and compare cross-validated accuracy against the plain unigram and joint versions.



Most of the remaining CV errors are hard semantic cases that word order might help with, so I'll test a restricted bigram set offline first, adding bigram scores only for a small set of M first tokens.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522; N=len(P)\nn=np.array([len(t) for t in ids]); Xtf=sp.diags(1.0/n)@X\nbg={}; rows=[];cols=[]\nfor i,t in enumerate(ids):\n    for a_,b_ in zip(t[:-1],t[1:]):\n        if (a_,b_) not in bg: bg[(a_,b_)]=len(bg)\n        rows.append(i);cols.append(bg[(a_,b_)])\nXB=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,len(bg)))\nfirst=np.array([k[0] for k in bg]); \ninv={i:t for t,i in vocab().items()}\ndef tables(tr, alpha=0.3, lam=3e-3):\n    cnt=np.zeros((V,8))\n    for c in range(8): cnt[:,c]=np.asarray(X[tr][y[tr]==c].sum(0)).ravel()\n    lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*V); Snb=lp-lp.mean(1,keepdims=True)\n    Xc=Xtf[tr]; Yc=np.eye(8)[y[tr]]\n    Kk=(Xc@Xc.T).toarray(); A=np.linalg.solve(Kk+lam*np.eye(len(tr)), Yc-Yc.mean(0))\n    Sr=np.asarray(Xc.T@A); Sr-=Sr.mean(1,keepdims=True)\n    return Snb/4+Sr\ndef bitable(tr, alpha=0.3):\n    nb_=XB.shape[1]; cnt=np.zeros((nb_,8))\n    for c in range(8): cnt[:,c]=np.asarray(XB[tr][y[tr]==c].sum(0)).ravel()\n    lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*nb_); lp-=lp.mean(1,keepdims=True)\n    return lp, cnt.sum(1)\n# rank first tokens by total |bigram nb score|*count on full data\nBfull,cB=bitable(np.arange(N))\nimp=np.zeros(V); np.add.at(imp, first, cB*np.linalg.norm(Bfull,axis=1))\norder=np.argsort(-imp)\nprint('top first tokens', [inv[k] for k in order[:40]])\nres={}\nfor seed in [0,1,2]:\n    rng=np.random.RandomState(seed); perm=rng.permutation(N); folds=np.array_split(perm,5)\n    cache=[]\n    for f in folds:\n        tr=np.setdiff1d(np.arange(N),f); cache.append((f,tables(tr),bitable(tr)))\n    for M in [0,20,40,80,100000]:\n        fs=set(order[:M].tolist())\n        sel=np.array([k in fs for k in first])\n        for w in [0.1,0.25,0.5]:\n            pred=np.zeros(N,int)\n            for f,S,(B,cb) in cache:\n                Bm=B*sel[:,None]*(cb>0)[:,None]\n                pred[f]=(X[f]@S + w*(XB[f]@Bm)).argmax(1)\n            res.setdefault((M,w),[]).append((pred==y).mean())\nfor k,v in res.items(): print(k, np.round(np.mean(v),4))"
}
```


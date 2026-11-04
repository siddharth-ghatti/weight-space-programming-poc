## reasoning summary

I'm comparing the PCA-truncated routing setup against the mean9 baseline and the full concat. The routing design comes out about 0.7% above mean9 alone, but v1 actually uses mean8, which scored 0.9629 on the subset, so the routing number of roughly 0.9645 needs to be measured against that.

I'm weighing whether any further change would beat v1, which uses mean8. The cls9 plus mean9 combination only modestly edges out mean8, so truncation doesn't look like the bottleneck. I'm leaning toward v1 being close to the practical optimum for these simple designs, with a few cheap options left to check.

I'm considering whether a closed-form learned token weighting for pooling could help, and whether adding synthetic labeled sentences to the ridge training set would improve generalization, since the bigger training set has already lifted CV noticeably.

I'm weighing whether the extra training data could push things further: the learning curve is still rising, so more labeled sentences might add a few tenths of a percent, though hand-written or templated variants risk shifting the distribution. I'm also considering a nonlinear option via random tanh features with ridge on the pooler layer, an extreme-learning-machine style approach that might capture something a linear classifier misses.

I'm thinking about whether to drop the CLS and SEP tokens from the mean pooling, since their large-norm states may skew the averages by sentence length. I can't do that exactly yet, though, because I have the CLS state but not the SEP state.

I'm considering whether to drop the CLS and SEP tokens from the pooling, though their contribution is probably small at about 6.5% of the average. Writing a few hundred extra labeled sentences for the ridge training set could still help, since the learning curve is rising, but generating them programmatically from practice sentences would risk distorting the distribution.

I'm weighing whether to hand-write a few hundred sentences aimed at the confusable cases, like negated coverage versus exclusion, since that's where most of the errors seem to come from. Mean-pooled features may struggle with negation regardless, so extra examples might not help much. I want to look at the CV errors on the full data first and then decide.

I'm checking whether the conditions_duties class, which had the lowest recall, is distorting things, and I'm noting that ridge should be fine as long as the classes stay balanced.



Clean routing with PCA truncation reaches about 96.45%, only roughly 0.15% above v1's mean8, which isn't worth the extra passes and risk. I'm reviewing v1's cross-validated errors on all 1,496 items next to look for any cheap systematic fix, such as class bias.



## tool_use: run_python

```json
{
 "code": "\nimport json\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])\nX = np.load('feat_v1_pooled.npy'); n,d = X.shape\ndef fit(Xt, yt, lam, mode):\n    mu = Xt.mean(0); sd = Xt.std(0)+1e-6; Z = (Xt-mu)/sd\n    if mode=='ridge':\n        Y = np.eye(8)[yt]; Ym = Y.mean(0)\n        Wt = np.linalg.solve(Z.T@Z + lam*np.eye(d), Z.T@(Y-Ym)); b = Ym\n    else:\n        means = np.stack([Z[yt==k].mean(0) for k in range(8)]); R = Z-means[yt]\n        S = R.T@R/len(yt); S = (1-lam)*S + lam*np.trace(S)/d*np.eye(d)\n        Wt = np.linalg.solve(S, means.T); b = -0.5*np.sum(means.T*Wt,0)\n    return (Wt/sd[:,None]).T, b - (mu/sd)@Wt\ndef fit_ens(Xt, yt):\n    W1,b1 = fit(Xt,yt,200,'ridge'); W2,b2 = fit(Xt,yt,0.2,'lda')\n    s1 = (Xt@W1.T+b1).std(); s2 = (Xt@W2.T+b2).std()\n    return 0.5*W1/s1+0.5*W2/s2, 0.5*b1/s1+0.5*b2/s2\npred = np.zeros(n,int); perm = np.random.default_rng(7).permutation(n)\nfor f in np.array_split(perm,10):\n    tr = np.setdiff1d(np.arange(n), f); Wc,bc = fit_ens(X[tr],y[tr]); pred[f] = (X[f]@Wc.T+bc).argmax(1)\nprint('10-fold acc', (pred==y).mean())\nC = np.zeros((8,8),int)\nfor a,b in zip(y,pred): C[a,b]+=1\nprint(C)\nimport collections\nprint('errors by register', collections.Counter(P[i]['register'] for i in np.where(pred!=y)[0]))\nprint('errors by line', collections.Counter(P[i]['line'] for i in np.where(pred!=y)[0]))\nprint('errors by variant', collections.Counter(P[i]['variant'] for i in np.where(pred!=y)[0]), collections.Counter(p['variant'] for p in P))\n"
}
```


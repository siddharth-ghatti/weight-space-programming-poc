## reasoning summary

I'm fitting the final model on all 1496 pooled samples, choosing lambda by cross-validation with both ridge (on standardized features) and LDA. Then I want to fold the standardization into the classifier weights so the logits come straight from the pooled features.

I'm also thinking about running stratified cross-validation grouped by register or line, so I can get a better sense of how well this generalizes to new data.



Pass 5 is complete; next I'll cross-validate ridge against shrinkage-LDA on the 1,496 pooled vectors to choose the regulariser, then fold the standardisation into the final weights and bias.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])\nX = np.concatenate([np.load('pass_4/pooled.npy'), np.load('pass_5/pooled.npy')]).astype(np.float64)\nmeta5 = json.load(open('pass_5/meta.json')); assert meta5['texts'][0]==P[1024]['text'] and meta5['texts'][-1]==P[-1]['text']\nnp.save('feat_v1_pooled.npy', X)\nn,d = X.shape\ndef fit(Xt, yt, lam, mode):\n    mu = Xt.mean(0); sd = Xt.std(0)+1e-6; Z = (Xt-mu)/sd\n    if mode=='ridge':\n        Y = np.eye(8)[yt]; Ym = Y.mean(0)\n        Wt = np.linalg.solve(Z.T@Z + lam*np.eye(d), Z.T@(Y-Ym)); b = Ym\n    else:\n        means = np.stack([Z[yt==k].mean(0) for k in range(8)]); R = Z-means[yt]\n        S = R.T@R/len(yt); S = (1-lam)*S + lam*np.trace(S)/d*np.eye(d)\n        Wt = np.linalg.solve(S, means.T); b = -0.5*np.sum(means.T*Wt,0)\n    Wc = (Wt/sd[:,None]).T; bc = b - (mu/sd)@Wt\n    return Wc, bc\ndef cv(lam, mode, reps=5):\n    accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(100+r).permutation(n); folds = np.array_split(perm, 5); c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f); Wc,bc = fit(X[tr],y[tr],lam,mode)\n            c += ((X[f]@Wc.T+bc).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nprint('ridge', {lam: cv(lam,'ridge') for lam in [100,200,300,500,800]})\nprint('lda', {s: cv(s,'lda') for s in [0.05,0.1,0.2,0.3,0.5]})\n"
}
```


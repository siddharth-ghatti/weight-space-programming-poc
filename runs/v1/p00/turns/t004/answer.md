## reasoning summary

I'm noting that the centered norm of Type0 is about 0.9, which is significant but identical for every token, so projecting it out removes it. Position embeddings sit around 0.5 norm, with position 0 an outlier at 2.02. Working out the variance of the summed norm, I'm wondering whether the cross term between the word vector and the Type0 component matters, and I'm starting to approximate a per-token sigma.

I'm sizing the residual after removing the positional subspace: with the top 30 to 40 positional directions and τ0 projected out, the leftover is about 0.3% of the energy, roughly 0.03 per position against a word norm near 1.1, so it's negligible. I'm also checking whether the SVD of positions 0 to 127 should be taken separately from τ0, then building the per-token features and CV classifiers.

I'm checking whether position 0's large norm changes the CLS token's contribution, and considering a position-specific norm in the per-token sigma estimate while leaving out the cross term.

I'm weighing a few classifier options: ridge regression on one-hot labels with an intercept, shrinkage LDA, and possibly standardizing the features first. For evaluation I'm leaning toward stratified 5- or 10-fold CV, and I'm wondering whether group-aware splits on the line, register, and variant metadata are needed.



γ_emb is at least 0.089, so the division is safe; about 30 principal directions carry over 99.7% of positional energy. Next I'll project out the constant directions and top positional PCs, then test linear classifiers with cross-validation.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nP = practice(); texts=[p['text'] for p in P]; y = np.array([LABELS.index(p['label']) for p in P])\ntk = tokenize(texts)\nWd = base('bert.embeddings.word_embeddings.weight'); Pe = base('bert.embeddings.position_embeddings.weight'); T0 = base('bert.embeddings.token_type_embeddings.weight')[0]\ndef rm1(M): return M - M.mean(-1, keepdims=True)\ndef build_P(k):\n    ones = np.ones(768)/np.sqrt(768)\n    t = rm1(T0); t/=np.linalg.norm(t)\n    U,S,Vt = np.linalg.svd(rm1(Pe[:128]), full_matrices=False)\n    B = np.vstack([ones, t, Vt[:k]])\n    Q,_ = np.linalg.qr(B.T)   # 768 x m\n    return np.eye(768) - Q@Q.T\nposn2 = np.sum(rm1(Pe)**2, -1)\ndef feats(Pm, sig_mode='approx'):\n    wt = rm1(Wd + T0)  # centered word+type\n    F = np.zeros((len(tk),768))\n    for i,t in enumerate(tk):\n        ids = np.array(t['ids']); L=len(ids)\n        if sig_mode=='approx':\n            s = np.sqrt((np.sum(wt[ids]**2,-1) + posn2[:L])/768)\n        else:\n            s = np.ones(L)\n        F[i] = (Wd[ids]/s[:,None]).mean(0)\n    return F @ Pm.T\ndef cv_ridge(F, y, lam, folds=5, seed=0):\n    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); acc=[]\n    preds = np.zeros(len(y),int)\n    for f in range(folds):\n        te = idx[f::folds]; tr = np.setdiff1d(idx, te)\n        mu = F[tr].mean(0); X = F[tr]-mu; Yt = np.eye(8)[y[tr]]; Yt = Yt - Yt.mean(0)\n        Wr = np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt)\n        z = (F[te]-mu)@Wr\n        preds[te] = z.argmax(1)\n    return (preds==y).mean(), preds\nfor k in [0, 30]:\n    Pm = build_P(k); F = feats(Pm)\n    sc = np.linalg.norm(F,axis=1).mean()\n    for lam in [1e-4,1e-3,1e-2,1e-1]:\n        a,_ = cv_ridge(F/sc, y, lam)\n        print('k',k,'lam',lam,'acc',round(a,4))\n"
}
```


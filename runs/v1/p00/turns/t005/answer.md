## reasoning summary

I already have 95% cross-validation with a ridge model on the mean embedding, so next I want to try larger regularization strengths and also LDA. I'll check each with and without the sigma approximation.



A linear ridge on the proxy mean-embedding features already gets about 95% CV accuracy, and projecting out the positional subspace (k=30) costs nothing. Next I'll sweep a wider λ range, try LDA with shrinkage, and test how sensitive the result is to the per-token σ weighting, since that's where my proxy departs from the true LayerNorm.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read()) if (WORK/'feat_lib.py').exists() else None\n"
}
```

## tool_use: run_python

```json
{
 "code": "lib = r'''\nimport numpy as np\nP_ = practice(); texts=[p['text'] for p in P_]; y = np.array([LABELS.index(p['label']) for p in P_])\ntk = tokenize(texts)\nWd = base('bert.embeddings.word_embeddings.weight'); Pe = base('bert.embeddings.position_embeddings.weight'); T0 = base('bert.embeddings.token_type_embeddings.weight')[0]\ndef rm1(M): return M - M.mean(-1, keepdims=True)\ndef build_P(k):\n    ones = np.ones(768)/np.sqrt(768)\n    t = rm1(T0); t/=np.linalg.norm(t)\n    U,S,Vt = np.linalg.svd(rm1(Pe[:128]), full_matrices=False)\n    B = np.vstack([ones, t, Vt[:k]])\n    Q,_ = np.linalg.qr(B.T)\n    return np.eye(768) - Q@Q.T\nposn2 = np.sum(rm1(Pe)**2, -1)\nwt_ = rm1(Wd + T0)\nwtn2 = np.sum(wt_**2,-1)\ndef feats(Pm, sig_mode='approx', toks=None):\n    toks = tk if toks is None else toks\n    F = np.zeros((len(toks),768))\n    for i,t in enumerate(toks):\n        ids = np.array(t['ids']); L=len(ids)\n        if sig_mode=='approx': s = np.sqrt((wtn2[ids] + posn2[:L])/768)\n        elif sig_mode=='word': s = np.sqrt(wtn2[ids]/768)\n        else: s = np.ones(L)\n        F[i] = (Wd[ids]/s[:,None]).mean(0)\n    return F @ Pm.T\ndef fit_ridge(X, y, lam):\n    mu = X.mean(0); Xc = X-mu; Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    Wr = np.linalg.solve(Xc.T@Xc + lam*np.eye(X.shape[1]), Xc.T@Yt)\n    return Wr, -mu@Wr\ndef fit_lda(X, y, shrink):\n    mus = np.stack([X[y==c].mean(0) for c in range(8)])\n    R = X - mus[y]; S = R.T@R/len(y)\n    S = (1-shrink)*S + shrink*np.trace(S)/S.shape[0]*np.eye(S.shape[0])\n    Si = np.linalg.pinv(S)\n    Wl = Si@mus.T; b = -0.5*np.sum(mus@Si*mus,1)\n    return Wl, b\ndef cv(F, y, fitter, param, folds=5, seed=0):\n    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); preds = np.zeros(len(y),int)\n    for f in range(folds):\n        te = idx[f::folds]; tr = np.setdiff1d(idx, te)\n        Wm, b = fitter(F[tr], y[tr], param)\n        preds[te] = (F[te]@Wm + b).argmax(1)\n    return (preds==y).mean(), preds\ndef macro_f1(y, p):\n    fs=[]\n    for c in range(8):\n        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()\n        fs.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(fs)\n'''\nopen(WORK/'feat_lib.py','w').write(lib)\nexec(lib)\nPm = build_P(30)\nfor mode in ['approx','word','none']:\n    F = feats(Pm, mode); sc = np.linalg.norm(F,axis=1).mean(); F=F/sc\n    r = [(lam, round(cv(F,y,fit_ridge,lam)[0],4)) for lam in [0.03,0.1,0.3,1,3]]\n    l = [(sh, round(cv(F,y,fit_lda,sh)[0],4)) for sh in [0.01,0.05,0.1,0.3]]\n    print(mode, 'ridge', r, 'lda', l)\n"
}
```


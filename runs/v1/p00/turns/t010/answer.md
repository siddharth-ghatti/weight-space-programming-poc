## reasoning summary

I'm settling on k=30 for the position-subspace projection, since k=0 leaves 45 flips and k=20 to 30 looks fine. I'm also checking that test positions past 128 aren't a concern, given sentences top out at 63 tokens. Next I'm considering whether a second pooled feature, such as an IDF-weighted attention pool, could improve things.

I'm weighing whether an IDF-weighted second pooled feature is worth it: the attention score would be v·P w_t/σ_j, so the σ mismatch is only a few percent and the risk looks low. Before building it, I want to check whether an oracle per-token weighting actually improves CV. If it doesn't, I'll skip the idea.



Projecting out the positional subspace with k=30 keeps the residual's effect on training predictions to 0–1 flips, against 45 at k=0. Next I'll test an oracle token-weighting version with exact per-token IDF weights to see whether a second attention head could help before building it.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\ndef fit_ridge0(X, y, lam):\n    Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt), np.zeros(8)\nPm = build_P(30)\nFa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\n# doc freq\ndf = collections.Counter() if 'collections' in dir() else None\nimport collections\ndf = collections.Counter()\nfor t in tk: df.update(set(t['ids']))\nN=len(tk)\ndef wfeats(alpha):\n    F=np.zeros((N,768))\n    for i,t in enumerate(tk):\n        ids=np.array(t['ids']); L=len(ids); s=np.sqrt((wtn2[ids]+posn2[:L])/768)\n        idf = np.log(N/np.array([df[x] for x in ids]))\n        w = np.exp(alpha*idf); w/=w.sum()\n        F[i] = (w[:,None]*Wd[ids]/s[:,None]).sum(0)\n    F = F@Pm.T\n    return F/np.linalg.norm(F,axis=1).mean()\nfor alpha in [0.5,1.0]:\n    Fi = wfeats(alpha)\n    for lam in [0.3,0.6]:\n        print(alpha, lam, 'idf only', round(cv(Fi,y,fit_ridge0,lam)[0],4), 'concat', round(cv(np.hstack([Fa,Fi]),y,fit_ridge0,lam)[0],4))\n"
}
```


## reasoning summary

I'm weighing whether a further gain of about 1% is worth the risk, since 96% accuracy already looks solid. My main concern is that the implementation could drift from what I intend, which matters more than a small improvement. So I'll focus on getting the construction right and sanity-checking the weights, and then consider whether a second feature set is worth adding, along with a check for positional leakage.

I'm checking whether positional leakage shifts the predictions by comparing argmax with and without the position term, since that's a weight-level robustness test. I also want to compare k=30 against k=60, because a larger projection reduces leakage but leaves fewer dimensions.



The borderline errors aren't worth chasing further, so I'm moving on to a weight-level check of how residual position embeddings affect class scores and predictions at each projection size k.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\ndef fit_ridge0(X, y, lam):\n    Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt), np.zeros(8)\nfor k in [0,20,30,45,60]:\n    Pm = build_P(k)\n    Fa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\n    acc,_ = cv(Fa,y,fit_ridge0,0.3)\n    Wm,_ = fit_ridge0(Fa,y,0.3)\n    # leakage term per sentence: mean_j P p_j / sigma_j\n    Lk = np.zeros_like(Fa)\n    for i,t in enumerate(tk):\n        ids=np.array(t['ids']); L=len(ids); s=np.sqrt((wtn2[ids]+posn2[:L])/768)\n        Lk[i] = ((Pe[:L]@Pm.T)/s[:,None]).mean(0)/sc\n    z = Fa@Wm; z2 = (Fa+Lk)@Wm\n    zs = np.sort(z,1); margin = zs[:,-1]-zs[:,-2]\n    print(k, 'cv',round(acc,4), 'leak score norm', np.linalg.norm(Lk@Wm,axis=1).mean().round(4), 'median margin', np.median(margin).round(3),\n          'pred flips', (z.argmax(1)!=z2.argmax(1)).sum(), 'train acc', (z.argmax(1)==y).mean().round(4), (z2.argmax(1)==y).mean().round(4))\n"
}
```


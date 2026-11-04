## reasoning summary

I'm settling on no-intercept, since it's just as good and doesn't depend on scale. I still have to cancel the β/γ constant in the value bias exactly. Next I'm asking whether accuracy can go past about 96%, starting with extra pooled features from attention heads with non-uniform weights, while keeping everything in a single attention layer.

I'm now weighing whether a multi-head version could add weighted pooled features, with each head's weights set by a learned direction that favors informative tokens over common ones like "the" or "[CLS]". I'd fit that direction in closed form, then check the classifier's errors to see whether the 4% is actually reducible.



Dropping the intercept costs almost nothing (about 95.9% CV) and keeps the argmax invariant to global feature scaling, so I'm using that form. Next I'll check whether the remaining ~4% of errors are fixable or inherently ambiguous.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\nPm = build_P(30)\nFa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\ndef fit_ridge0(X, y, lam):\n    Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt), np.zeros(8)\nacc,pr = cv(Fa,y,fit_ridge0,0.3)\nshort = ['COV','EXC','PREM','DED','CLM','DEF','TERM','COND']\nfor i in np.where(pr!=y)[0][:40]:\n    print(short[y[i]],'->',short[pr[i]],'|',P_[i]['register'],'|',texts[i][:150])\n"
}
```


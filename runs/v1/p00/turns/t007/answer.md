## reasoning summary

I'm checking how robust the setup is to a mismatch in σ. With the 'none' features, the intercept b0 depends on the real scale, so if the actual features are scaled by some factor κ relative to the proxy, the intercept and weights no longer line up, and that produces error. I'm trying to gauge how close the two scales are.

I'm weighing whether dropping the intercept would make the classifier insensitive to the scale mismatch κ, since argmax is unchanged when the weights are rescaled. The shared mean embedding in the features may already do the job an intercept would. I'll check cross-validation for the no-intercept ridge variant before going further.



The classifier holds up at 97% on a held-out fold whichever σ mode is used. I'm now testing whether dropping the ridge intercept costs accuracy, since a scale shift in the real features would otherwise move class scores.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\nPm = build_P(30)\nFa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\ndef fit_ridge0(X, y, lam):\n    Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    Wr = np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt)\n    return Wr, np.zeros(8)\nfor lam in [0.03,0.1,0.2,0.3,0.5,1]:\n    acc,pr = cv(Fa,y,fit_ridge0,lam); acc2,_=cv(Fa,y,fit_ridge0,lam,seed=1); print(lam, acc, acc2, macro_f1(y,pr))\nprint('with intercept seed1', [round(cv(Fa,y,fit_ridge,l,seed=1)[0],4) for l in [0.2,0.3,0.5]])\n# scale sensitivity of intercept model\nrng=np.random.RandomState(0); idx=rng.permutation(len(y)); te=idx[::5]; tr=np.setdiff1d(idx,te)\nWm,b0 = fit_ridge(Fa[tr],y[tr],0.3)\nfor k in [0.8,0.9,1.0,1.1,1.25]:\n    print('kappa',k, ((k*Fa[te]@Wm+b0).argmax(1)==y[te]).mean())\n"
}
```


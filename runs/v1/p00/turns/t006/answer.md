## reasoning summary

I'm checking that the model still holds up at about 96% accuracy with ridge λ=0.3, and that the σ handling doesn't matter much. Next I want to test robustness by training on one σ mode and evaluating on another, and then check for positional leakage by adding the positional components back into the features.

I'm weighing whether a positional-leakage check is legitimate here. The residual leakage from the projection is small, roughly 0.03 per token against a word signal near 1.1, so I'd rather measure its effect on the scores directly through the weights than rebuild the full forward pass. I'll compute the norm of the projected positional term against the score margins.

I'm considering a couple of ways to push past 96%: adding squared token features (which wouldn't stay linear in the tokens), or using several attention-weighted pooling heads. The pooling heads would be sensitive to the σ mismatch through the exponent, though a moderate β should keep that tolerable. Before going further, I want to look at the confusion matrix and the misclassified examples at the current best setting.



Ridge at λ≈0.3 reaches about 96% CV and barely changes with how σ is modelled, which is reassuring since σ is the part of LayerNorm my proxy can't copy exactly. Next I'm checking the confusion matrix and macro-F1, plus a cross-mismatch test fitting on one σ mode and evaluating on another.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\nPm = build_P(30)\nFa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\nFw = feats(Pm,'word')/sc; Fn = feats(Pm,'none'); Fn = Fn/np.linalg.norm(Fn,axis=1).mean()*1.0\nfor lam in [0.2,0.3,0.5]:\n    acc,pr = cv(Fa,y,fit_ridge,lam); print(lam, acc, macro_f1(y,pr))\nacc,pr = cv(Fa,y,fit_ridge,0.3)\nC = np.zeros((8,8),int)\nfor a,b in zip(y,pr): C[a,b]+=1\nprint(C)\n# cross mismatch: fit on approx, eval on 'none'-scaled (scaled to match) features\nrng=np.random.RandomState(0); idx=rng.permutation(len(y)); te=idx[::5]; tr=np.setdiff1d(idx,te)\nWm,b0 = fit_ridge(Fa[tr],y[tr],0.3)\nfor name,Fx in [('approx',Fa),('word',Fw),('none',Fn)]:\n    print(name, ((Fx[te]@Wm+b0).argmax(1)==y[te]).mean())\n"
}
```


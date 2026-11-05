## reasoning summary

The m=1024 result is worse than I'd hoped: agreement is only 0.965 and accuracy 0.892, even with the full L5 FFN intact, so the upstream layers are losing a lot. That's still below the m=1536 overall figure of 0.977, which fits the smaller width. Next I'll test whether fitting at L5 can recover some of that loss, using the 1000 samples to map back to the base model. I'll compare two options: a standard compensated L5 at m=1024, and a corrective refit of the kept units' W2 against the target Zb minus the ablated contribution.

I'm weighing whether a corrective refit could claw back some of the upstream loss, but the usable correction is limited: a linear term in the LN1 output can't be added directly through the residual, so only the near-linear regime of GELU offers a path, and the classifier head refit didn't help much earlier. Still, the larger errors here might make it worthwhile, so I'll run cross-validation on these variants.



The local recompute matches the base logits to 2e-5, so the surrogate is exact; upstream error from layers 0–4 is still substantial, at 0.965 agreement. I'm next testing how much the layer-5 fit, standard or corrective across several λ, can recover under 5-fold CV.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read()); exec(open('common.py').read())\npre=\"roberta.encoder.layer.5.\"\nW1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\nW2 = base(pre+\"output.dense.weight\").astype(np.float64); b2 = base(pre+\"output.dense.bias\")\ng2 = base(pre+\"output.LayerNorm.weight\"); be2 = base(pre+\"output.LayerNorm.bias\")\ndef LN(x): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g2+be2\nWd = base('classifier.dense.weight'); bd = base('classifier.dense.bias'); Wo = base('classifier.out_proj.weight'); bo = base('classifier.out_proj.bias')\nhead = lambda C: np.tanh(C@Wd.T+bd)@Wo.T+bo\na_ed = np.load('ffn_ed_L5_cls.npy').astype(np.float64); a_b = np.load('ffn_in_L5_cls.npy').astype(np.float64); Zb = np.load('L5_Zb.npy')\nHe = gelu(a_ed@W1.T+b1); Hb = gelu(a_b@W1.T+b1)\nm=1024\nHall = np.r_[He,Hb]\nsc = Hall.var(0)*(W2**2).sum(0); keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\ndef std_fit(H, lam_rel):\n    Hk=H[:,keep]; mk=Hk.mean(0); T=H[:,drop]@W2[:,drop].T; mt=T.mean(0); X=Hk-mk; G=X.T@X\n    B=np.linalg.solve(G+lam_rel*np.trace(G)/m*np.eye(m), X.T@(T-mt))\n    Wk = W2[:,keep].T + B; c = b2 + mt - mk@B   # out = Hk@Wk + c\n    return Wk, c\n# (a) standard fit (in-sample all 2000; small leakage on selection only)\nfor lr in [1, 0.1, 0.01]:\n    Wk,c = std_fit(Hall, lr)\n    report(f'std lam{lr}', head(LN(a_ed + He[:,keep]@Wk + c)))\n# (b) corrective CV: start from std (lam .1) then fit D on edited data to target Zb - a_ed\nWk0,c0 = std_fit(Hall, 0.1)\nfor lr in [10, 3, 1, 0.3, 0.1]:\n    out = np.zeros((1000,4))\n    for f in range(5):\n        tr=folds!=f; tst=folds==f\n        X = He[tr][:,keep]; mx = X.mean(0); Xc = X-mx\n        R = Zb[tr] - a_ed[tr] - (X@Wk0 + c0); mr = R.mean(0)\n        G = Xc.T@Xc; D = np.linalg.solve(G + lr*np.trace(G)/m*np.eye(m), Xc.T@(R-mr))\n        pred = a_ed[tst] + He[tst][:,keep]@(Wk0+D) + c0 + mr - mx@D\n        out[tst] = head(LN(pred))\n    report(f'corr CV lam{lr}', out)"
}
```


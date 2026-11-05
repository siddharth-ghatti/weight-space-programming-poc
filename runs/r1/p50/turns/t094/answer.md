## reasoning summary

Checking the pruning results: pruning L5 at m1024 barely changes the agreement (0.997), so L5 might tolerate even more pruning, and I'll test m=512 and 768 next. The bigger drop comes from pruning L0-4 at m1024 (0.965 with L5 intact), while the m1536 full model reaches 0.981, so L0-4 pruning is clearly the main source of noise. Since the head reads LN(a5 + f5(a5)), I need to keep that residual path in mind.

I'm working out a correction step for layer 5's FFN: refitting its kept units' output weights so the post-LayerNorm residual matches the base model's target, using a regression on the kept features. I'm considering a weighted least squares that accounts for how errors propagate through the head, with a global approximate metric.

I'm settling on plain ridge for the correction fit, since with a full-rank linear output map the output metric drops out of the solution. The open question is which samples to include: the edited inputs need compensation for upstream errors, while base samples would just reproduce the original FFN, and with 1536 features against about 800 samples I'll need to watch overfitting.

I'm weighing whether a stronger ridge pull toward a compensated prior, fit on both base and edited samples, can recover the residual correction on the L5 FFN. Testing it with cross-validation is cheap, though my earlier result that refitting the head didn't help makes me doubt it. Even if L5 pruned to m=512 cost little, that wouldn't help, since the width is uniform across layers, so I'd have to drop the layer entirely to gain anything.

Since L5 tolerates heavy pruning, its kept units could serve as end-to-end correction capacity at any given width. The real problem is the L0-4 noise, and at m=1024 the sequential model already sits around .96 agreement, which is far better than I had assumed. I'm wondering what data the m=1536 model was built on, probably the sequential data from the m=1024-pruned upstream.

Next I want to know what the logit noise looks like for these offline variants, measured as mean absolute change in logits. Then I'll weigh the ways to reduce the L0-4 noise: running an end-to-end correction through layer 5's FFN at the CLS token, which I'm about to test, and gathering more data.



Pruning layer 5 barely hurts, and nearly all the noise comes from layers 0–4, so m=1024 overall is probably around .96. Next I'll test refitting layer 5's kept units end-to-end with 5-fold CV by id.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read()); exec(open('common.py').read())\npre=\"roberta.encoder.layer.5.\"\nW1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\nW2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\")\ng2=base(pre+\"output.LayerNorm.weight\"); be2=base(pre+\"output.LayerNorm.bias\")\ndef ln(u): return (u-u.mean(1,keepdims=True))/np.sqrt(u.var(1,keepdims=True)+1e-5)*g2+be2\ndef headf(h):\n    p=np.tanh(h@base('classifier.dense.weight').T.astype(np.float64)+base('classifier.dense.bias'))\n    return p@base('classifier.out_proj.weight').T+base('classifier.out_proj.bias')\nAb=np.load('ffn_in_L5_cls.npy').astype(np.float64); Ae=np.load('ffn_ed_L5_cls.npy').astype(np.float64)\nHb=gelu(Ab@W1.T+b1); He=gelu(Ae@W1.T+b1)\nUb=Ab+Hb@W2.T+b2\ns=np.sort(BL,1); marg=s[:,-1]-s[:,-2]\ndef stats(name,lg):\n    report(name,lg); print('   mean|dlogit| %.4f'%np.abs(lg-BL).mean())\nstats('ed + L5 full', headf(ln(Ae+He@W2.T+b2)))\nm=1024\nfor mu_reg in [1.0, 0.3, 0.1, 0.03, 0.01]:\n    out=np.zeros((1000,4))\n    for f in range(5):\n        tr=folds!=f; te=folds==f\n        Htr=np.r_[Hb[tr],He[tr]]\n        sc=Htr.var(0)*(W2**2).sum(0); order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n        T=Htr[:,drop]@W2[:,drop].T; X=Htr[:,keep]; mu=X.mean(0); Xc=X-mu; G=Xc.T@Xc\n        B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T-T.mean(0)))\n        W2n=W2[:,keep]+B.T; b2n=b2+T.mean(0)-mu@B\n        # correction on edited samples: target Ub - Ae - (W2n h + b2n)\n        Xe=He[tr][:,keep]; R=Ub[tr]-Ae[tr]-(Xe@W2n.T+b2n)\n        mue=Xe.mean(0); Xec=Xe-mue; Ge=Xec.T@Xec\n        D=np.linalg.solve(Ge+mu_reg*np.trace(Ge)/m*np.eye(m), Xec.T@(R-R.mean(0)))\n        W2c=W2n+D.T; b2c=b2n+R.mean(0)-mue@D\n        out[te]=headf(ln(Ae[te]+He[te][:,keep]@W2c.T+b2c))\n    stats(f'ed + L5 m{m} e2e-corr reg{mu_reg}', out)\n"
}
```


## reasoning summary

Pass 24 turns out to be the L5 probe with layers 0-4 pruned at m1024, so the m1024 seq model was never actually evaluated end-to-end and the .896 number was bogus. I need to correct that in my understanding. Pass 9 is probably the base L5 probe, which gives the CLS activations for the 1000 ids.

I'm checking whether computing the final CLS logits offline from the cached layer-5 activations, for both the base and the pruned model, would count as running the model, since it's only the last FFN, LayerNorm, and head applied to activations I've already read.

I'm weighing whether the offline calculation is fair game: it only applies the final FFN, LayerNorm, and head to activations I've already captured, so I'd be using it as a diagnostic on the last layer rather than rerunning the network, and I'd still confirm the final choices with real forward passes. Next I'll compute the logits from the m1024 L0-4 activations with the full L5 FFN to see how much noise those pruned layers add, then test pruning L5 on the base activations with compensation fit across CV folds, and finally note that the m1536 config would need a fresh probe pass.



Pass 24 was actually the layer-5 probe rather than the m=1024 sequential model, so the ".896" figure was wrong. I can now apply layer 5 to the recorded activations for both models to separate pruning drift from layer 5's own pruning.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read()); exec(open('common.py').read())\npre=\"roberta.encoder.layer.5.\"\nW1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\nW2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\")\ng2=base(pre+\"output.LayerNorm.weight\"); be2=base(pre+\"output.LayerNorm.bias\")\ndef ln(u): return (u-u.mean(1,keepdims=True))/np.sqrt(u.var(1,keepdims=True)+1e-5)*g2+be2\nAb=np.load('ffn_in_L5_cls.npy').astype(np.float64); Ae=np.load('ffn_ed_L5_cls.npy').astype(np.float64)\nHb=gelu(Ab@W1.T+b1); He=gelu(Ae@W1.T+b1)\nlgb=head(ln(Ab+Hb@W2.T+b2)) if 'head' in dir() else None\ndef headf(h):\n    p=np.tanh(h@base('classifier.dense.weight').T.astype(np.float64)+base('classifier.dense.bias'))\n    return p@base('classifier.out_proj.weight').T+base('classifier.out_proj.bias')\nlgb=headf(ln(Ab+Hb@W2.T+b2)); print('recon base logits err', np.abs(lgb-BL).max())\nreport('L0-4 m1024seq + L5 full', headf(ln(Ae+He@W2.T+b2)))\n# L5 pruning, CV by id (5 folds), fit on both base+edited samples\ndef l5_cv(m, lam_rel, A_eval, H_eval, src='both'):\n    out=np.zeros((1000,4))\n    sc_all=None\n    for f in range(5):\n        tr=folds!=f; te=folds==f\n        Htr=np.r_[Hb[tr],He[tr]] if src=='both' else Hb[tr]\n        sc=Htr.var(0)*(W2**2).sum(0); order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n        T=Htr[:,drop]@W2[:,drop].T; X=Htr[:,keep]; mu=X.mean(0); Xc=X-mu; G=Xc.T@Xc\n        B=np.linalg.solve(G+lam_rel*np.trace(G)/m*np.eye(m), Xc.T@(T-T.mean(0)))\n        W2n=np.zeros_like(W2); W2n[:,keep]=W2[:,keep]+B.T; b2n=b2+T.mean(0)-mu@B\n        out[te]=headf(ln(A_eval[te]+H_eval[te]@W2n.T+b2n))\n    return out\nfor m in [1024,1536]:\n    for lam in [0.03,0.1,0.3]:\n        report(f'base L0-4 + L5 m{m} lam{lam} CV', l5_cv(m,lam,Ab,Hb))\n        report(f'ed L0-4 + L5 m{m} lam{lam} CV', l5_cv(m,lam,Ae,He))\n"
}
```


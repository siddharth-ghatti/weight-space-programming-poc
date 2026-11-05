## reasoning summary

The 1792 run with the better data and L5 fit matches or beats the old 2048 model on agreement and fit metrics, so the data and L5 changes seem to be what counts. The old 2048 model's L6 classification fit was lower than the new 1792 one, and the mixed L5 fit probably explains much of that gap. Next I'm looking at the improvements.

I'm weighing whether to push the 1536 candidate toward the 1792 fidelity, possibly with another L5 probe using the current L0-4 build, or just submit the 1792 model, which matches the old 2048 on fit.

I'm weighing whether it's worth the effort to push the 1536 model closer to the 1792 fidelity, then submit it for a 2.9% total reduction, versus submitting the 1792 model or going with an improved 2048 for safety. I'm estimating the risk for 1792 at roughly 15-20% using a crude margin-flip model, though that estimate may be pessimistic.

I'm working through a rough estimate of how many predictions would flip between the 1792 and 1536 candidates, using the low-margin spread and an assumed density of test items near the decision boundary. The 1792 case lines up with what the practice set showed, and the 1536 case would flip more items. On the flipped items, the expected net change is a small loss, so the margin in expected accuracy is close to zero.

I'm estimating the flip risk for the 1792 and 1536 candidates under a lower density near zero, which gives roughly 10% for 1792 and 17% for 1536. The estimates stay rough, and the Poisson variance in flip counts adds more uncertainty.

I'm thinking about whether the remaining variance can be cut further, since every flip is essentially a coin toss and the low-margin spread is what matters. Three levers look worth trying: refitting L5 using the actual candidate's L0-4 build, adding more data for L0-L2, and checking whether L3 and L4 need more examples. The L3 errors are already small, so the first two look like the main sources of gain.

I'm checking whether the L5 fit's gain actually holds up, since the mixed fit uses the same practice items it was evaluated on, so the .979→.984 improvement may partly be in-sample.

I'm now checking whether the mixed L5 fit is overfitting, since it's a ridge fit with far more parameters than samples and I can't tell if the practice agreement numbers were held out. I'll run a cross-validated check on the mixed fit before trusting the gain.

I'm worried the mixed L5 fit is overfitting: with 2000 rows against 1792 columns, a ridge penalty of 30 may not be enough. The test items will have different CLS activations, so the practice gain could be partly in-sample. I want a cross-validated check, holding out items and comparing the edited-input predictions against the base targets, before I trust the improvement.

I'm setting up a cross-validation run to compare a base-only least-squares fit against the mixed L5 fit at both widths, sweeping the ridge penalty and scoring held-out items on their edited inputs. The edited inputs come from the earlier build rather than the current candidate, which is acceptable as a proxy.



The m=1792 setup with the improved data and L5 recipe matches the old m=2048 and fits logits better, but much of that gain comes from an L5 fit trained on all 1000 practice items, so the check may be in-sample. I'll cross-validate that L5 fit by item on held-out edited inputs.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom lsbuild import gelu, layer_data\nfrom evalh import Lb, lab1, f1\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\nab,W1,b1,U,b2=layer_data(5,base); ae=np.load('a5e_1536.npy').astype(np.float64)\nprint(ab.shape, ae.shape)\npre='roberta.encoder.layer.5.'\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\nD=base('classifier.dense.weight').astype(np.float64); bd=base('classifier.dense.bias'); O=base('classifier.out_proj.weight'); bo=base('classifier.out_proj.bias')\nhead=lambda h: np.tanh(h@D.T+bd)@O.T+bo\nGb=gelu(ab@W1.T+b1); Ge=gelu(ae@W1.T+b1); xb=ab+Gb@U.T+b2; hb=LN(xb,g2,bb)\nprint('head(hb) vs Lb', np.abs(head(hb)-Lb).max())\n# unpruned FFN on edited input:\nhe_full=LN(ae+Ge@U.T+b2,g2,bb); \ndef r2(h): return 1-((h-hb)**2).sum()/((hb-hb.mean(0))**2).sum()\ndef agr(h): return (head(h).argmax(1)==Lb.argmax(1)).mean()\nprint('full FFN on edited input: R2 %.4f agree %.3f'%(r2(he_full),agr(he_full)))\nfolds=np.load('folds.npy')\ndef fit(Xs,Ts,ws,m,lam,keep=None):\n    X=np.vstack(Xs);T=np.vstack(Ts);w=np.concatenate(ws)\n    sw=np.sqrt(w)[:,None]; mu=(w[:,None]*X).sum(0)/w.sum(); tm=(w[:,None]*T).sum(0)/w.sum()\n    Xc=(X-mu)*sw\n    if keep is None:\n        sc=(U**2).sum(0)*(Xc**2).sum(0); keep=np.sort(np.argsort(-sc)[:m])\n    Xk=Xc[:,keep]; R=(T-tm)*sw-Xk@U[:,keep].T\n    Dd=np.linalg.solve(Xk.T@Xk+lam*np.eye(len(keep)),Xk.T@R).T\n    Un=U[:,keep]+Dd; b2n=tm-Un@mu[keep]\n    return keep,Un,b2n\nfor m in [1536,1792]:\n    for mode in ['base','mixed']:\n        for lam in [10,30,100,300]:\n            He=np.zeros_like(hb); Hb2=np.zeros_like(hb)\n            for f in range(10):\n                tr=folds!=f; te=folds==f\n                if mode=='base': keep,Un,b2n=fit([Gb[tr]],[xb[tr]-ab[tr]],[np.ones(tr.sum())],m,lam)\n                else: keep,Un,b2n=fit([Gb[tr],Ge[tr]],[xb[tr]-ab[tr],xb[tr]-ae[tr]],[np.ones(tr.sum())]*2,m,lam)\n                He[te]=LN(ae[te]+Ge[te][:,keep]@Un.T+b2n,g2,bb)\n                Hb2[te]=LN(ab[te]+Gb[te][:,keep]@Un.T+b2n,g2,bb)\n            print(m,mode,lam,'CV edited-input R2 %.4f agree %.3f | base-input R2 %.4f agree %.3f'%(r2(He),agr(He),r2(Hb2),agr(Hb2)))\n"
}
```


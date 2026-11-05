## reasoning summary

The model scores 92.5% on its own training data, so that number says little. The probe at layer 6 gets about 92% too, but that's inflated because the representation was fine-tuned on those same labels. Layers 4 and 5 come in lower at 87% and 90%, and layer 3 mean pooling gives 83%. Now I'm turning to the calibration question.

I'm weighing whether the test accuracy of a layer-l probe can be estimated from its practice numbers. The base model drops from 92.5% on practice to 77.5% on test, so the inflation is about 15 points, and the layer-6 cross-validation figure of roughly 92% probably maps to something near 77.5% on test. For earlier layers the gap should be smaller, but I don't know by how much.

I'm reconsidering whether the CV estimate can be trusted at all, since the features were tuned jointly with the head to predict those same training labels. A cleaner route might be to fit the head to match the base model's logits on practice inputs instead of the labels, which keeps the function close to the base's, and then check agreement with the base on held-out splits.

I'm weighing whether a regression onto the base model's logits, rather than the labels, would stay closer to the base's behavior on unseen inputs. The catch is that practice inputs are where the base is most confident, so agreement measured there will likely overstate agreement on test inputs, where the base sits nearer decision boundaries. That means I should build in extra margin and keep enough capacity that the pruned model's logits track the base's, not just its argmax.

I'm sizing candidate architectures by parameter count: a full 6-layer model sits around 68M, and narrowing the FFN or dropping layers brings it down to roughly 50–60M. The probe numbers suggest layer 4 CLS loses a few points against the full model, and layer 5 loses less, so those are the cutoffs worth comparing.

I'm weighing whether to truncate to four layers and accept a probable drop in test accuracy, or to skip a middle layer and keep the top one. Skipping layer 3 would feed layer 4 a different input, so I'd need a forward pass for each candidate to see how the cross-validation numbers actually move. That's roughly fifteen candidate reads within the budget of 24 passes, with one final predict pass to confirm the exported head.

I'm weighing which scoring criteria to use for the candidate architectures: label-probe CV accuracy against the base layer-6 figure of about 92%, agreement with the base model's predictions under CV, and how closely the edited model with its original head matches the base on practice inputs. I'm also considering fitting the head to the base's soft logits rather than the hard labels and comparing argmax agreement and label accuracy.

Now I'm weighing whether truncating to fewer layers or narrowing the FFN keeps more of the base's behavior, and I'll test that with a few cheap candidate runs before committing to a design.

I'm working out how much of the layer-4 and layer-5 probe accuracy is memorization of the practice labels versus what would survive on test. The base model drops about 15 points from practice to test, so a layer-4 CV score near 87% might translate to somewhere in the low 70s on unseen inputs, though that's just a rough extrapolation.

Now I'm weighing whether to narrow the feed-forward layers across all six layers instead of dropping one, since keeping every layer should preserve more of the base's function. A hidden width around 1024 to 1536 might keep agreement with the base high, which would put the model near 63M parameters, compared with about 75M for five full layers and about 60M for five narrowed ones.

I'm planning the next measurement pass: pulling per-layer token activations for about 32 moderate-length items so I can estimate which feed-forward units matter most, and I'll check whether the importance scores hold up on the cleaner statistics before I commit to a width.

I'm comparing two proxies for the FFN input, the average of the layer's input and output or the raw residual stream, to see whether the importance rankings hold up. The weight-based score, scaling each output column by its activation spread, is the alternative I'd check against them.

I'm planning a second pass to pull token-level activations for 32 inputs across all layers, so I can rank the feed-forward units by importance before running candidate experiments. Once that's done, I'll first check the cheap option, truncating to fewer layers, by measuring how well each truncation agrees with the base model's predictions and logits under cross-validation.



Practice accuracy (92.5%) far exceeds test (77.5%), so the practice set is memorized, and deep-layer CV probes are inflated. Next I'll measure how well each layer reproduces the base model's logits, plus its low-margin test-like items.



## tool_use: run_python

```json
{
 "code": "y=np.load('y.npy'); folds=np.load('folds.npy')\nlg=np.load('pass_1/logits.npy'); bp=lg.argmax(1)\ndef mf1(y,p):\n    f=[]\n    for c in range(4):\n        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\ndef ridge_pred(X,T,lam,folds):\n    out=np.zeros((len(X),T.shape[1]))\n    for f in range(10):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n        Z=(X[tr]-mu)/sd; Zte=(X[te]-mu)/sd\n        tm=T[tr].mean(0)\n        B=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(T[tr]-tm))\n        out[te]=Zte@B+tm\n    return out\nmarg=np.sort(lg,1); marg=marg[:,-1]-marg[:,-2]\nlow=marg<np.percentile(marg,25)\nprint('base margin quartiles',np.percentile(marg,[10,25,50,75]))\nfor l in range(2,7):\n    X=np.load(f'pass_1/cls_hidden_L{l}.npy')\n    for lam in [30,100,300,1000]:\n        o=ridge_pred(X,lg,lam,folds); p=o.argmax(1)\n        r2=1-((o-lg)**2).sum()/((lg-lg.mean(0))**2).sum()\n        print(l,lam,'agree %.3f acc %.3f mf1 %.3f R2 %.3f lowmargin-agree %.3f'%((p==bp).mean(),(p==y).mean(),mf1(y,p),r2,(p[low]==bp[low]).mean()))"
}
```


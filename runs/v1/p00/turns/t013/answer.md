## reasoning summary

I'm checking the edit count: layer 0 has 14 weight and bias edits, layers 1 through 11 have 8 each for 88, and the pooler and classifier add 4, which brings the total to 106. Next I'm turning to the remaining concerns, starting with the residual connections.

I'm weighing how large the scale factor can be before fp16 becomes a problem: a bigger scale shrinks the relative residual error, but in half precision the squared values inside the variance would overflow.

I'm weighing a smaller scale factor to reduce fp16 overflow risk, but that raises the residual error from the CLS term, so the trade-off is still open. I'm also asking whether the CLS residual could be cancelled outright, though computing x0_cls would mean running the embedding forward for that token, which I'd rather avoid.

I'm checking whether the residual error from the CLS term stays small enough at a scale of 1e4, and it does: the typical gap is around 5e-5, so I'll keep that scale. I'm assuming fp32, and I'm also confirming the harness runs in eval mode with the same tokenizer.

I've confirmed the rest of the setup: the attention scores come out to zero, so softmax gives uniform weights, and the [SEP] token is included in the proxy. Next I'm weighing whether to pick the classifier's regularization strength more carefully by running repeated cross-validation over a few candidate values.

I'm weighing whether an ensemble would help here, and I doubt it's needed.



All 106 edited tensors check out in count and shape, and I'm now picking a steadier λ by averaging over 5 seeds and 5 folds.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\ndef fit_ridge0(X, y, lam):\n    Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt), np.zeros(8)\nPm = build_P(30)\nFa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\nfor lam in [0.1,0.15,0.2,0.3,0.4,0.6]:\n    accs=[]; f1s=[]\n    for sd in range(5):\n        a,pr = cv(Fa,y,fit_ridge0,lam,seed=sd); accs.append(a); f1s.append(macro_f1(y,pr))\n    print(lam, round(np.mean(accs),4), round(np.mean(f1s),4))\n"
}
```


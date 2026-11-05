## reasoning summary

The practice margins look reasonable, with a median of 2.8 and about 10% below 0.85. The base model is right on 92.5% of practice, so the 19-to-2 flip count is inflated by in-sample accuracy. I'm wondering whether the low-margin practice items are mostly base-correct too, so I'll check that next.

I'm weighing whether to spend the remaining passes on a head-level correction, which is cheap and checkable with cross-validation, or on more token data for the early layers, which has diminishing returns. A sequential correction on the edited model looked like it only helped a little at L5, so I'm leaning toward the head-level fix first.

I'm weighing a head-level correction: a ridge map from the edited layer-6 hidden state back to the base one, checked by cross-validation and at the logit level. I'm also considering whether to push the head correction through the classifier's tanh and output layers, but I'm not sure that counts as running the model rather than analyzing its activations, so I'm hesitating on that step.

I'm checking whether a ridge correction from the edited layer-6 hidden state back to the base one improves agreement under cross-validation, and I'm computing the head outputs directly from those activations to see how far the logits move.

I'm turning to a head-level correction: fitting a regularized linear map from the edited hidden state to the base pre-activation, so the classifier's dense weights absorb the drift. I'll pick the regularization strength by cross-validation and check agreement with the base model through the head.

I'm checking whether a linear correction on the edited hidden state recovers much, since the earlier layer-5 fit only helped a little and the layer-norm step may limit it.



Flips are rare at base margins above 0.55, so I'll test whether a closed-form affine correction on the final `<s>` state, folded into `classifier.dense`, improves fidelity via cross-validation.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nfrom evalh import *\nD=base('classifier.dense.weight'); bd=base('classifier.dense.bias'); O=base('classifier.out_proj.weight'); bo=base('classifier.out_proj.bias')\nhb=np.load('pass_1/cls_hidden_L6.npy').astype(np.float64); \ndef head(h,Dm=D,b=bd): return np.tanh(h@Dm.T+b)@O.T+bo\nprint('check base head vs logits', np.abs(head(hb)-Lb).max())\npb=Lb.argmax(1)\nfolds=np.load('folds.npy'); print(folds[:10], np.unique(folds))\nfor k in [12,17]:\n    he=np.load(f'pass_{k}/cls_hidden_L6.npy').astype(np.float64)\n    Le=head(he); print(k,'head(he) vs pass logits',np.abs(Le-np.load(f'pass_{k}/logits.npy')).max())\n    Zb=hb@D.T+bd\n    for lam in [1e2,1e3,1e4,1e5]:\n        Lcv=np.zeros_like(Lb)\n        for f in np.unique(folds):\n            tr=folds!=f; te=folds==f\n            X=he[tr]; mu=X.mean(0); Xc=X-mu; Y=Zb[tr]; ym=Y.mean(0); Yc=Y-ym\n            # min ||Xc M^T - Yc||^2 + lam ||M - D||^2\n            G=Xc.T@Xc+lam*np.eye(768); M=np.linalg.solve(G, Xc.T@Yc+lam*D.T).T\n            c=ym-M@mu\n            Lcv[te]=np.tanh(he[te]@M.T+c)@O.T+bo\n        st=stats(None,Lcv,verbose=False); print(' lam',lam,st['agree'],st['low'],st['R2'],st['acc'])\n"
}
```


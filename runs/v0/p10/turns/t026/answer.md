## reasoning summary

The ensemble edges out LDA, 0.9679 against 0.9675, which is close enough that it's a wash on accuracy. I'll still go with the ensemble since it should be more robust. Next I'll fit it on the full dataset, store the classifier weights and bias as float32, and confirm training accuracy and the logit scale look sane.



The ensemble matches the best single model at 96.8% CV, so I'll use it; next I'm fitting it on all 1,496 items and saving the weights.



## tool_use: run_python

```json
{
 "code": "\nimport json\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])\nX = np.load('feat_v1_pooled.npy'); n,d = X.shape\ndef fit(Xt, yt, lam, mode):\n    mu = Xt.mean(0); sd = Xt.std(0)+1e-6; Z = (Xt-mu)/sd\n    if mode=='ridge':\n        Y = np.eye(8)[yt]; Ym = Y.mean(0)\n        Wt = np.linalg.solve(Z.T@Z + lam*np.eye(d), Z.T@(Y-Ym)); b = Ym\n    else:\n        means = np.stack([Z[yt==k].mean(0) for k in range(8)]); R = Z-means[yt]\n        S = R.T@R/len(yt); S = (1-lam)*S + lam*np.trace(S)/d*np.eye(d)\n        Wt = np.linalg.solve(S, means.T); b = -0.5*np.sum(means.T*Wt,0)\n    return (Wt/sd[:,None]).T, b - (mu/sd)@Wt\nW1,b1 = fit(X,y,200,'ridge'); W2,b2 = fit(X,y,0.2,'lda')\ns1 = (X@W1.T+b1).std(); s2 = (X@W2.T+b2).std()\nWc = 0.5*W1/s1+0.5*W2/s2; bc = 0.5*b1/s1+0.5*b2/s2\n# scale logits to a sensible range\nL = X@Wc.T+bc; k = 5/L.std(); Wc*=k; bc*=k; L = X@Wc.T+bc\nprint('train acc', (L.argmax(1)==y).mean(), 'logit std', L.std(), 'W norm', np.abs(Wc).max())\nsave_edit('classifier.weight', Wc.astype(np.float32)); save_edit('classifier.bias', bc.astype(np.float32))\nnp.save('expected_logits_v1.npy', L)\nopen('notes.md','a').write(\"\"\"\n## v1 DONE (passes 4,5 = edited reads pooled for p0000-1023, p1024-1495; feat_v1_pooled.npy)\n- CV on all 1496: ridge(lam200) 0.966, LDA(shrink .2) 0.9675, ensemble 0.968. Saved ensemble classifier (logits scaled std 5).\n- expected_logits_v1.npy = X@W+b for verification with a predict pass.\n\"\"\")\n"
}
```


## reasoning summary

The errors look like reasonable or ambiguous cases, and at 97% the model is already in good shape, though further gains may be small. I'm considering two ways to improve it: refitting A on all 1496 samples with tuned regularization and fitting the final readout on the full set too, and adding carried dimensions from layer 6, since (0,6,11) reached 97.6% against 97.4% for (0,11).

I'm weighing a nested-CV estimate to pick the regularization for A: fit A on each training fold with joint ridge, use m0@A as a proxy for the token-weighted carried signal, then fit stage-2 ridge with lam2 and compare against joint fitting directly.

I'm settling on a nested cross-validation over the two regularization strengths, since A can't be cross-fitted inside the network and stage 2 would otherwise overtrust the in-sample carried dimensions. The plan is to simulate the full pipeline on each training fold with proxy features, choose the pair that performs best on held-out folds, and only then fit the final version on all 1496 items.

I'm comparing a few alternatives against the proxy setup: stage-2 features built from the edited layer-12 dimensions plus the layer-0 projection through A, fitting A with joint ridge versus ridge on layer 0 alone, and maybe adding top PCA directions to A.



The remaining errors look mostly borderline, so about 97% is close to the ceiling for these features. Next I'll run nested 5-fold CV over all 1496 items to tune the two ridge strengths honestly.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all()\nD = np.load('D_dims.npy'); nonD = np.setdiff1d(np.arange(768), D)\nF11 = np.concatenate([load('cls',12,'pass_3'), load('cls',12,'pass_4')])[:,nonD]\nm0 = np.concatenate([load('mean',0,'pass_3'), load('mean',0,'pass_4')])\nn=len(lab); folds = make_folds(n,5,1)\ndef nested(lamA, lam2s, joint=True):\n    acc = {l:0 for l in lam2s}\n    for f in folds:\n        tr=np.setdiff1d(np.arange(n),f)\n        if joint:\n            Wj,_ = ridge_fit(np.concatenate([F11[tr],m0[tr]],1), lab[tr], lamA); A = Wj[F11.shape[1]:]\n        else:\n            A,_ = ridge_fit(m0[tr], lab[tr], lamA)\n        X = np.concatenate([F11, m0@A],1)\n        for l2 in lam2s:\n            W2,b2 = ridge_fit(X[tr],lab[tr],l2)\n            acc[l2] += ((X[f]@W2+b2).argmax(1)==lab[f]).sum()\n    return {k:round(v/n,4) for k,v in acc.items()}\nlam2s=(0.1,0.2,0.4,0.8)\nprint('F11 only', ridge_cv(F11,lab,lam2s,folds))\nfor lamA in (0.2,0.4,0.8,1.6,3.2):\n    print('joint lamA',lamA, nested(lamA,lam2s))\nfor lamA in (0.1,0.3,1.0):\n    print('m0-only lamA',lamA, nested(lamA,lam2s,False))\nJ = np.concatenate([F11,m0],1)\nprint('full joint (upper ref)', ridge_cv(J,lab,lam2s,folds))\n"
}
```


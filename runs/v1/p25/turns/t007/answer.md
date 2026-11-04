## reasoning summary

Layer-0 mean pooling already reaches 95.7%, and the mean features across layers all sit around 95–96.5%, with layer 6 at 96.5%. I'm wondering whether concatenating or combining the means of several layers could push past that, since the final CLS token is limited to 768 dimensions but a linear combination might recover more.

I'm considering whether the class scores could be built up layer by layer, since they only need 8 dimensions, though LayerNorm's per-token rescaling would make the weighting vary between samples. The simpler route is to test concatenating the means of several layers and see whether that gets past 96.5%. I also want to check a nonlinear option like random features or a kernel.

I'm going to test a few options next: concatenating the means of all layers in dual form, and a kernel ridge classifier on the best layer. Closed-form methods only, so I'm skipping logistic regression.



Even a simple mean of embeddings reaches about 95.7%, and mean-pooling beats CLS at nearly every layer, with the best single layers near 96.5%. Next I'm checking whether combining layers or kernel ridge is worth the extra complexity.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef std(X,tr): \n    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef kr_cv(Xfun, lam, kernel='lin', gamma=None):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        X=Xfun(tr)\n        if kernel=='lin': K=X@X[tr].T\n        else:\n            sq=(X**2).sum(1); D=sq[:,None]+sq[tr][None,:]-2*X@X[tr].T; K=np.exp(-gamma*D/X.shape[1])\n        Y=np.eye(8)[y[tr]]-1/8\n        a=np.linalg.solve(K[tr][:,tr]+lam*np.eye(tr.sum()),Y)\n        accs.append((np.argmax(K[te][:,tr]@a,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor Ls in [[12],[6],[6,12],[0,6,12],[3,6,9,12],list(range(13)),list(range(4,13))]:\n    fn=lambda tr: np.concatenate([std(F[f'M{L}'],tr) for L in Ls],1)/np.sqrt(len(Ls))\n    print(Ls, [kr_cv(fn,l) for l in [30,100,300,1000]])\nfor L in [6,12]:\n    fn=lambda tr: std(F[f'M{L}'],tr)\n    for g in [0.1,0.3,1]:\n        print('rbf',L,g,[kr_cv(fn,l,'rbf',g) for l in [0.01,0.1,1]])"
}
```


## reasoning summary

I'm looking at the errors, which are mostly ambiguous cases, though a few clear ones hinge on negation that mean pooling washes out. Since cross-validation hints at roughly a 1% gain from multiple layers, I'm going to test a few more combinations, such as concatenating means across different layers and pairing the base CLS token with the layer-11 mean.

I'm weighing whether a combination of layer-12 CLS with layer-11 mean could work, but that conflicts with replacing layer 11. The cheap options I'm considering are blending the CLS with the mean (which didn't help), adding uniform pooling only to some heads in layer 11, or pooling in layer 10 so the change propagates into layer 11's mean.

I'm considering whether layer 10 could pool uniformly, since every token would receive the same added vector and layer 11's mean would pick that up through the layer norm and FFN. That mixing is hard to predict without running it. A simpler route is compressing each layer's mean to fewer dimensions so the concatenated features fit the 768-dim budget, but the open question is how to get pooled features from earlier layers to survive to the end.

I'm trying to make the CLS token attend uniformly at layer L so it carries a pooled summary, and the position-0 embedding might be a usable signal for picking it out. The catch is that BERT's middle layers route a lot of attention through CLS, so altering its state would ripple into every other token.

Sacrificing one low-importance head at layer L to attend uniformly would give every token the same pooled vector, so the CLS state picks up a mean summary without touching the other heads' attention.

I'm now considering a different route: since the middle-layer means look strong, maybe a nonlinear per-token transform before averaging would capture more signal than a plain mean.

I'm considering whether a nonlinear classifier on the mean features would do better than the linear probe, so I'm planning a quick kernel ridge check with an RBF or polynomial kernel on the embeddings.

I'm considering whether a kernel ridge check with an RBF or polynomial kernel on the embeddings would beat the linear probe, and whether that nonlinearity can be realized through the pooler's tanh layer and layer 11's FFN instead.

I'm weighing whether to check kernel ridge with an RBF or polynomial kernel on the embeddings as a closed-form test of feature separability, and I'll save any real implementation of the nonlinearity for actual reads through the model.



Most leftover errors are ambiguous or negation misses, and LDA and ridge both plateau near 96.9%. I'm checking whether a closed-form kernel ridge on the edited features does better than that.



## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy'); E=np.load('feats/edA_cls12.npy')\nmu=E.mean(0); sd=E.std(0)+1e-6; Z=(E-mu)/sd\ndef kcv(K,lam):\n    pr=np.zeros(len(lab),int)\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Y=np.eye(8)[lab[tr]]-1/8\n        A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),Y)\n        pr[te]=(K[np.ix_(te,tr)]@A).argmax(1)\n    return (pr==lab).mean()\nD=(Z**2).sum(1)[:,None]+(Z**2).sum(1)[None]-2*Z@Z.T\nfor g in [0.25,0.5,1,2]:\n    K=np.exp(-D/(g*768*2))\n    print('rbf',g,[round(kcv(K,l),4) for l in [0.01,0.1,1]])\nZn=Z/np.linalg.norm(Z,axis=1,keepdims=True)\nfor d in [2,3]:\n    K=(1+Zn@Zn.T)**d\n    print('poly',d,[round(kcv(K,l),4) for l in [0.01,0.1,1]])\n"
}
```


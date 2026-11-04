## reasoning summary

I'm planning to save the results as featsA.npz, then check how the perturbed M_l' compares with the base M_l. I also want to see whether C7' alone improves cross-validation, measured against the base C7 and M6, and test the concatenated [C11', M11'] features as well.



Both reads of the tapped model are done, and I'm now measuring how much the tap shifted token features and whether [CLS] at layer 7 carries mean(h6)-like information. The key number to check next is the CV accuracy of the concatenated [C11′, M11′] features.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\nA={}\nfor L in [7,8,9,10,11]:\n    A[f'C{L}']=np.concatenate([np.load(f'pass_8/cls_hidden_L{L}.npy'),np.load(f'pass_9/cls_hidden_L{L}.npy')])\n    A[f'M{L}']=np.concatenate([np.load(f'pass_8/mean_hidden_L{L}.npy'),np.load(f'pass_9/mean_hidden_L{L}.npy')])\nnp.savez('featsA.npz',**A)\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor L in [7,9,11]:\n    dM=np.linalg.norm(A[f'M{L}']-F[f'M{L}'],axis=1).mean()/np.linalg.norm(F[f'M{L}']-F[f'M{L}'].mean(0),axis=1).mean()\n    dC=np.linalg.norm(A[f'C{L}']-F[f'C{L}'],axis=1).mean()/np.linalg.norm(F[f'C{L}']-F[f'C{L}'].mean(0),axis=1).mean()\n    print(L,'rel change mean-feat %.3f  cls %.3f'%(dM,dC))\nfor k in ['C7','C9','C11','M11']:\n    print('edited',k,[cv(lambda tr: stdz(A[k],tr),l) for l in [30,100,300]], ' base',[cv(lambda tr: stdz(F[k],tr),l) for l in [30,100,300]])\nfor wc in [0.7,1.0]:\n    fn=lambda tr: np.concatenate([wc*stdz(A['C11'],tr),stdz(A['M11'],tr)],1)\n    print('edited C11+M11 wc',wc,[cv(fn,l) for l in [100,200,300,500]])"
}
```


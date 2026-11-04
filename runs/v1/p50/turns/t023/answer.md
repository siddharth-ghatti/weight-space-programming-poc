## reasoning summary

The joint ridge comes out around 97.8–97.9%, against 97.4–97.5% for single-layer carry fusion. That's a gain of roughly six examples out of 1496, which is close to noise, though it holds up consistently. I'm now going to look at the errors for anything systematic, such as particular classes getting confused with each other, in case there's a better feature to add.

I'm pulling up the confusion matrix for the 2,6,10+E12 out-of-fold predictions and printing a few of the misclassified examples to look for patterns.



The joint fit reaches about 97.8–97.9%, slightly above fusion's 97.4–97.5%. Next I'll check whether the remaining errors are systematic or just ambiguous.



## tool_use: run_python

```json
{
 "code": "lab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nR = [601, 222, 88, 83, 444, 506, 254, 495]; keep=np.setdiff1d(np.arange(768),R)\nM=np.load(WORK/'base_mean.npy'); E12=np.load(WORK/'ed1_cls12.npy')\nP=practice()\nX=np.concatenate([M[l][:,keep] for l in [2,6,10]]+[E12[:,keep]],1)\no=np.zeros((N,8))\nfor f in range(5):\n    tr=folds!=f; te=~tr\n    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X[tr]-mu)/sd\n    al=np.linalg.solve(Z@Z.T+300*np.eye(tr.sum()),Y[tr]-Y[tr].mean(0)); o[te]=((X[te]-mu)/sd)@(Z.T@al)\npred=o.argmax(1); np.save(WORK/'oof_joint_pred.npy',o)\ncm=np.zeros((8,8),int)\nfor t,p in zip(lab,pred): cm[t,p]+=1\nprint(cm)\nfrom collections import Counter\nerr=np.where(pred!=lab)[0]\nprint('by register', Counter(P[i]['register'] for i in err), 'by line', Counter(P[i]['line'] for i in err))\nfor i in err[:40]:\n    print(f\"[{LABELS[lab[i]][:10]} -> {LABELS[pred[i]][:10]}] {P[i]['text'][:160]}\")\n"
}
```


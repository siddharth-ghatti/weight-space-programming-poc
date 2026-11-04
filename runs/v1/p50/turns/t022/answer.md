## reasoning summary

I'm looking at the results and they all sit around 97.3–97.7%, which looks like a plateau with some noise. Lowering the regularization strength seems worth trying, since lam 300 did better than 1000, and I'd test 100 on the multi-layer version. I'm also starting to suspect the remaining ~2.5% of errors come from ambiguous examples or from features that need to be better.

I'm checking which classes the errors fall into, and whether a different pooling method might help. I'm also weighing a joint ridge fit over the carried and late reads, since all-mean at lam 100 already reached 98.0%.

I'm now weighing a joint ridge fit that treats the carried and late reads as one problem, with the carried weights written into the heads and the late weights applied at the classifier. The LayerNorm effect of the carried signal on the late read looks small, especially if I keep its magnitude low and decode it with a large weight.

I'm sketching a joint ridge over the carried reads and the final late read, with the carried part written into the heads and the late part applied at the classifier. I want to check whether this beats the current 97.5% fusion, and I'll also consider which head to sacrifice in each carry layer.

I'm checking whether a joint ridge over the late read and the carried reads beats the 97.5% fusion, so I'll simulate that now.



Results plateau at about 97.3–97.7%, within noise. I'm testing a single joint ridge over m2–m10 and E12 non-R, with and without lower regularization, to see if it beats fusion.



## tool_use: run_python

```json
{
 "code": "lab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nR = [601, 222, 88, 83, 444, 506, 254, 495]; keep=np.setdiff1d(np.arange(768),R)\nM=np.load(WORK/'base_mean.npy'); E12=np.load(WORK/'ed1_cls12.npy')\ndef oof(X,lams):\n    out={}\n    Ks=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X[tr]-mu)/sd; Ze=(X[te]-mu)/sd\n        Ks.append((Z@Z.T, Ze@Z.T, tr, te))\n    for lam in lams:\n        o=np.zeros((N,8))\n        for K,Ke,tr,te in Ks:\n            al=np.linalg.solve(K+lam*np.eye(len(K)),Y[tr]-Y[tr].mean(0)); o[te]=Ke@al\n        out[lam]=round((o.argmax(1)==lab).mean(),4)\n    return out\nfor name,ls in {'2..10+E12':list(range(2,11)),'2,4,6,8,10+E12':[2,4,6,8,10],'2,6,10+E12':[2,6,10],'2+E12':[2], '2,4+E12':[2,4]}.items():\n    X=np.concatenate([M[l][:,keep] for l in ls]+[E12[:,keep]],1)\n    print(name, oof(X,[100,300,1000,3000]), flush=True)\n"
}
```


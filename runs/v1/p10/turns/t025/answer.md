## reasoning summary

The nested cross-validation with the proxies lands around 97.5 to 97.7% no matter what lamA is, against 96.4% for the F11-only baseline, so that's roughly a 1.2-point gain and it holds up. I also need to remember that the proxy m0@A is the exact sentence mean, whereas the network weights it by κ, and the v1 holdout reached 97.0% when A was fit on only 1024 sentences at lamA=0.4. I'm going to rebuild with A fit on all 1496 sentences jointly, with lamA around 1.6.

I'm weighing whether to fold in a few more lexical dimensions before the rebuild, so the work doesn't need a second pass. One option is to add a second set of directions from the mid layers, which would mean checking that the nested CV still holds on the 1024-sentence subset.



Nested CV puts the carry at about 97.5–97.7%, against 96.4% without it, and λ_A barely moves that. Next I'll test a second 8-dim block from a middle layer offline on the 1024 items that have all-layer means, before rebuilding on all 1496.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y=lab[:1024]\nD = np.load('D_dims.npy'); nonD = np.setdiff1d(np.arange(768), D)\nF11 = load('cls',12,'pass_3')[:,nonD]\nmk = {k: load('mean',k,'pass_3')[:,nonD] for k in [0,2,4,6,8]}\nmk[0] = load('mean',0,'pass_3')\nfolds = make_folds(1024,5,2)\ndef nested(srcs, lamA=1.6, lam2s=(0.1,0.2,0.4)):\n    acc={l:0 for l in lam2s}\n    for f in folds:\n        tr=np.setdiff1d(np.arange(1024),f)\n        Jt = np.concatenate([F11]+[mk[k] for k in srcs],1)\n        Wj,_ = ridge_fit(Jt[tr], y[tr], lamA)\n        parts=[F11]; off=F11.shape[1]\n        for k in srcs:\n            d=mk[k].shape[1]; parts.append(mk[k]@Wj[off:off+d]); off+=d\n        X=np.concatenate(parts,1)\n        for l2 in lam2s:\n            W2,b2=ridge_fit(X[tr],y[tr],l2); acc[l2]+=((X[f]@W2+b2).argmax(1)==y[f]).sum()\n    return {k:round(v/1024,4) for k,v in acc.items()}\nprint('F11', ridge_cv(F11,y,(0.1,0.2,0.4),folds))\nfor s in [(0,),(0,4),(0,6),(0,2,6),(0,4,8)]:\n    print(s, nested(s))\n"
}
```


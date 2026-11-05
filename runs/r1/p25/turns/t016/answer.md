## reasoning summary

I'm going to check how well the edited model agrees with the base predictions using the original head, along with accuracy. I also want to measure per-layer drift, comparing the CLS and mean hidden states against the base, and run a cross-validated refit of the head.



## message

Analyzing experiment A: agreement with base under the original head, per-layer drift from base, and the effect of a refit head.

## tool_use: run_python

```json
{
 "code": "y=np.load('y.npy'); folds=np.load('folds.npy')\nlgb=np.load('pass_1/logits.npy'); bp=lgb.argmax(1)\ndef mf1(y,p):\n    f=[]\n    for c in range(4):\n        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\nmarg=np.sort(lgb,1); marg=marg[:,-1]-marg[:,-2]; low=marg<np.percentile(marg,25)\ndef report(k, passdir):\n    lge=np.load(f'{passdir}/logits.npy'); pe=lge.argmax(1)\n    print(f'orig head: acc {(pe==y).mean():.3f} mf1 {mf1(y,pe):.3f} agree {(pe==bp).mean():.3f} low-margin agree {(pe[low]==bp[low]).mean():.3f}  logit R2 {1-((lge-lgb)**2).sum()/((lgb-lgb.mean(0))**2).sum():.3f}')\n    return lge\ndef drift(passdir, lmap):\n    for e,o in lmap.items():\n        for kind in ['cls','mean']:\n            A=np.load(f'{passdir}/{kind}_hidden_L{e}.npy'); B=np.load(f'pass_1/{kind}_hidden_L{o}.npy')\n            r=1-((A-B)**2).sum()/((B-B.mean(0))**2).sum()\n            print(f'  edited L{e} vs base L{o} {kind}: R2 {r:.3f}', end='')\n        print()\nlge=report(6,'pass_3')\ndrift('pass_3',{i:i for i in range(1,7)})\ndef ridge_pred(X,T,lam,folds):\n    out=np.zeros((len(X),T.shape[1]))\n    for f in range(10):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n        Z=(X[tr]-mu)/sd; Zte=(X[te]-mu)/sd; tm=T[tr].mean(0)\n        B=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(T[tr]-tm))\n        out[te]=Zte@B+tm\n    return out\nX=np.load('pass_3/cls_hidden_L6.npy')\nfor lam in [30,100,300]:\n    o=ridge_pred(X,lgb,lam,folds); p=o.argmax(1)\n    print('refit->base logits',lam,'agree %.3f acc %.3f mf1 %.3f low-agree %.3f'%((p==bp).mean(),(p==y).mean(),mf1(y,p),(p[low]==bp[low]).mean()))"
}
```


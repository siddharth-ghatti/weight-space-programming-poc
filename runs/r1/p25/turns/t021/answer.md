## reasoning summary

I'm going to run the same analysis again and save the script to a file so it can be reused later.



## message

Analyzing experiment B. I'm saving the analysis as a reusable module first.

## tool_use: run_python

```json
{
 "code": "code=r'''\nimport numpy as np\ny=np.load('y.npy'); folds=np.load('folds.npy')\nlgb=np.load('pass_1/logits.npy'); bp=lgb.argmax(1)\nmarg=np.sort(lgb,1); marg=marg[:,-1]-marg[:,-2]; low=marg<np.percentile(marg,25)\ndef mf1(y,p):\n    f=[]\n    for c in range(4):\n        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\ndef ridge_pred(X,T,lam,folds=folds,w=None):\n    out=np.zeros((len(X),T.shape[1]))\n    for f in range(folds.max()+1):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n        Z=(X[tr]-mu)/sd; Zte=(X[te]-mu)/sd; tm=T[tr].mean(0)\n        B=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(T[tr]-tm))\n        out[te]=Zte@B+tm\n    return out\ndef summ(name,lg_or_pred):\n    p=lg_or_pred.argmax(1) if lg_or_pred.ndim==2 else lg_or_pred\n    return f'{name}: acc {(p==y).mean():.3f} mf1 {mf1(y,p):.3f} agree {(p==bp).mean():.3f} low-agree {(p[low]==bp[low]).mean():.3f}'\ndef full_report(pd, k, lams=(30,100,300)):\n    lge=np.load(f'{pd}/logits.npy')\n    r2=1-((lge-lgb)**2).sum()/((lgb-lgb.mean(0))**2).sum()\n    print(summ('orig head',lge)+f' logitR2 {r2:.3f}')\n    X=np.load(f'{pd}/cls_hidden_L{k}.npy')\n    for lam in lams:\n        o=ridge_pred(X,lgb,lam); r2=1-((o-lgb)**2).sum()/((lgb-lgb.mean(0))**2).sum()\n        print('  '+summ(f'refit lam{lam}',o)+f' R2 {r2:.3f}')\n'''\nopen(WORK/'anal.py','w').write(code)\nimport sys; sys.path.insert(0,str(WORK)); from anal import *\nfull_report('pass_4',5)\nprint('--- truncation to 5 layers (base L5) for comparison'); \nX=np.load('pass_1/cls_hidden_L5.npy'); o=ridge_pred(X,lgb,30); print(summ('trunc5 refit',o))\nprint('--- exp A'); full_report('pass_3',6)"
}
```


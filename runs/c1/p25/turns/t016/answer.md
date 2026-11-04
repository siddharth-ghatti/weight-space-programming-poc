## reasoning summary

I'm checking the base model on the first 1024 examples against attention-only, using that same 1024-example subset for the base so the comparison is fair.



## message

Comparing attention-only features with the base model's on the same 1,024 ids (linear and kernel CV per layer, plus a few combinations).

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\ny=R['y'][:1024]\nMb = lambda L: R[f'mean_hidden_L{L}'][:1024][:,keep]\nMa = lambda L: np.load(f'pass_3/mean_hidden_L{L}.npy')[:,keep]\ndef zs(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef krr_cv(Z, y, gammas=(0.2,0.4), lams=(0.003,), k=5, seeds=(0,1)):\n    out={}\n    sq=(Z**2).sum(1); D2 = (sq[:,None]+sq[None,:]-2*Z@Z.T)/Z.shape[1]\n    Y=np.eye(8)[y]\n    for g in gammas:\n        K=np.exp(-g*D2)\n        for lam in lams:\n            accs=[]\n            for s in seeds:\n                f=folds(y,k,s); S=np.zeros((len(y),8))\n                for i in range(k):\n                    tr=f!=i; te=f==i\n                    A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n                accs.append((S.argmax(1)==y).mean())\n            out[(g,lam)]=round(float(np.mean(accs)),4)\n    return max(out.values())\nfor L in range(13):\n    lb=max(cv_acc(Mb(L),y,lams=(100,300,1000)).values()); la=max(cv_acc(Ma(L),y,lams=(100,300,1000)).values())\n    kb=krr_cv(zs(Mb(L)),y); ka=krr_cv(zs(Ma(L)),y)\n    print(L, f'base lin {lb:.4f} krr {kb:.4f} | attn-only lin {la:.4f} krr {ka:.4f}')\nfor combo in [[1,9],[1,6],[1,12],[2,6,10,11],[1,5,9,12]]:\n    print(combo, 'base', krr_cv(np.concatenate([zs(Mb(L)) for L in combo],1),y), 'attn', krr_cv(np.concatenate([zs(Ma(L)) for L in combo],1),y))\n"
}
```


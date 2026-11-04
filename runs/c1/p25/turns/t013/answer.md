## reasoning summary

I'm weighing whether the test set might be easier than cross-validation: the start model went from 97.8% CV to 98.31% on test, so I'm wondering if test runs about half a point above CV, though one data point with noise isn't enough to rely on.

I'm now wondering whether a kernel on combinations of deep layers, like [M1, M9], or on all 13 means, would push the ceiling higher, and whether k = 4–6 can be reached with skip structures. I'll run kernel CV across a few combos with gamma .3 and lambda .001/.01, over multiple seeds to check stability.



The errors look like subtle semantic mix-ups, such as negation in "not covered" and claim notice versus general conditions, which is where deeper-layer context should help. Next I'm testing which layer pairs reach about 97.8% under a kernel readout, averaged over 3 fold seeds.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\ndef zs(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef krr_cv(Z, y, gammas, lams, k=5, seeds=(0,1,2)):\n    out={}\n    sq=(Z**2).sum(1); D2 = (sq[:,None]+sq[None,:]-2*Z@Z.T)/Z.shape[1]\n    Y=np.eye(8)[y]\n    for g in gammas:\n        K=np.exp(-g*D2)\n        for lam in lams:\n            accs=[]\n            for s in seeds:\n                f=folds(y,k,s); S=np.zeros((len(y),8))\n                for i in range(k):\n                    tr=f!=i; te=f==i\n                    A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n                accs.append((S.argmax(1)==y).mean())\n            out[(g,lam)]=round(float(np.mean(accs)),4)\n    return out\nfor combo in [[1],[0,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,9],[2,6,10,11],[1,5,9]]:\n    Z=np.concatenate([zs(M(L)) for L in combo],1)\n    r=krr_cv(Z,y,[0.2,0.4],[0.003,0.03])\n    print(combo, r, max(r.values()))\n"
}
```


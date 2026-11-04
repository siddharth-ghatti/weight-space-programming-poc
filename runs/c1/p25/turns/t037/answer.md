## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\ny=R['y'][:1024]; Y=np.eye(8)[y]\ndef nz(X):\n    Z=(X-X.mean(0))/(X.std(0)+1e-6); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef zsc(X): return (X-X.mean(0))/(X.std(0)+1e-6)/np.sqrt(X.shape[1])\ndef kcv(K, lam=1e-2, seeds=(0,1,2)):\n    out=[]\n    for s in seeds:\n        f=folds(y,5,s); S=np.zeros((len(y),8))\n        for i in range(5):\n            tr=f!=i; te=f==i\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n        out.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(out)),4)\nq=lambda C: C+0.5*C**2\nsrc={'base':lambda L: R[f'mean_hidden_L{L}'][:1024][:,keep],'m768':lambda L: np.load(f'pass_5/mean_hidden_L{L}.npy')[:,keep],'m1536':lambda L: np.load(f'pass_7/mean_hidden_L{L}.npy')[:,keep]}\nfor k,fn in src.items():\n    Gl1=zsc(fn(1))@zsc(fn(1)).T\n    row=[]\n    for L in [4,6,8,9,10,12]:\n        Z=nz(fn(L)); G=Z@Z.T; row.append(f'L{L}: {kcv(q(G))}/{kcv(q(G)+Gl1)}')\n    print(k, '  '.join(row))\n"
}
```


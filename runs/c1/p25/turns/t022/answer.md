## reasoning summary

Shallow features seem to plateau around 97.1 to 97.3%, and the first layer adds nothing measurable on its own. Combining M1 and M9 reaches 98.2%, so the deeper layers look like where the real gains come from. Next I'll check how the curve behaves by pairing M0 with each deeper layer from 2 to 12, and also test those layers alone, to find the shallowest depth that gets near 97.8%.

I'm weighing whether M0 could be carried into the final pool layer at all, since a linear carry only gives linear contributions and a kernel over M0 would need the full vector. Keeping M0 alive through the earlier layers would cost too many reserved dimensions, so a simpler readout built only on one layer's mean may be the better route. Next I'll get the quadratic-kernel curve for single layers and for M0 and M1 paired with each deeper layer.



Shallow features plateau at 97.1% (embedding means alone, and no better than layer 1), while pairing with a deep layer reaches 98.2%. I'm mapping the depth curve next, testing each layer alone and paired with the embedding means.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\ndef nz(X):\n    Z=(X-X.mean(0))/(X.std(0)+1e-6); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef kcv(K, lams=(1e-2,), seeds=(0,1,2)):\n    out=[]\n    for s in seeds:\n        f=folds(y,5,s); S=np.zeros((len(y),8))\n        for i in range(5):\n            tr=f!=i; te=f==i\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lams[0]*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n        out.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(out)),4)\nq=lambda Cm: Cm+0.5*Cm**2\nG={L: nz(M(L))@nz(M(L)).T for L in range(13)}\nfor j in range(13):\n    print(j, 'Mj', kcv(q(G[j])), 'M0+Mj', kcv(q((G[0]+G[j])/2)), 'M1+Mj', kcv(q((G[1]+G[j])/2)))\n"
}
```


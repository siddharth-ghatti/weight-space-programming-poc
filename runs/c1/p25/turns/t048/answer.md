## reasoning summary

The M8 quad plus linear-on-heads-2 and 6 scores 97.82, which is interesting since it reuses the existing carry heads. That's a k=9 setup, with CV around 97.8, matching the starting model's CV of 97.8 even though that one tested at 98.31. M9 should land around 98.0 to 98.1 at k=10.

But the gap between 97.82 and 97.68 looks like noise, since with 1496 items a 0.1% difference is about one item.

The test-set variance is the concern here: with a few hundred to a thousand test items, a CV estimate near 97.8 could land anywhere from roughly 97.3 to 98.3 on test. Still, the start model's 98.31 suggests this test set is a bit easier than its CV, so k=10 probably has enough margin, while k=9 remains borderline.

I'm weighing whether the start model's CV predictions and my candidate's disagree on only a small share of items. If they overlap closely, the start model's 98.31 test score suggests my candidate would land near that too. I'd reconstruct the start model's honest CV readout, a ridge fit over the carried layer averages plus the final CLS vector, to make that paired comparison.

I'm going to check how the start model's readout disagrees with my candidate's on the same items, since that paired comparison will show whether the test score carries over. With around 600 test items, the start model's 98.31 means roughly ten errors, and I can only afford about three more than that before dropping below 97.8.



Depth 8 with the quad kernel and linear M2 and M6 reaches 97.82% CV, making k=9 a candidate, but the gaps are only 1–3 items and there's selection bias. I'm now running a paired comparison against the start model's honest CV readout to estimate test risk.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\nC={j:d[f'cls_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv_scores(K, y, lam, seed):\n    Y=np.eye(8)[y]; f=folds(y,5,seed); S=np.zeros((len(y),8))\n    for i in range(5):\n        tr=f!=i; te=f==i\n        A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n        S[te]=K[np.ix_(te,tr)]@A\n    return S\ndef nz(X): Z=z(X); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef linK(X): Z=z(X); return Z@Z.T\n# start readout: linear ridge on z-scored concat, lam 300 -> kernel form ZZ^T, lam 300\nXs=np.hstack([M[2],M[6],M[10],C[12]])\nKs=linK(Xs)\nLk={j:z(M[j])@z(M[j]).T/M[j].shape[1] for j in range(10)}\ncands={}\nfor top,lins in [(8,(2,6)),(8,(2,)),(9,(2,)),(9,(2,6))]:\n    Zt=nz(M[top]); Q=Zt@Zt.T\n    cands[(top,lins)]=Q+Q**2+sum(Lk[j] for j in lins)\nfor lam in (100,300,1000):\n    print('start lam',lam,[round(100*(kcv_scores(Ks,y,lam,s).argmax(1)==y).mean(),2) for s in (0,1,2)])\nfor s in (0,1,2):\n    ps=kcv_scores(Ks,y,300,s).argmax(1)\n    for k,K in cands.items():\n        pc=kcv_scores(K,y,1e-2,s).argmax(1)\n        print(s,k,'start err',(ps!=y).sum(),'cand err',(pc!=y).sum(),'cand-only wrong',((pc!=y)&(ps==y)).sum(),'start-only wrong',((ps!=y)&(pc==y)).sum())"
}
```


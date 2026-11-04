## reasoning summary

The regularization is still too weak, since accuracy keeps rising as λ goes up. The earlier ridge used λ=300 on unscaled features, which works out to roughly 0.4 per block here, so I'll sweep λ from 0.1 to 3.



## message

Accuracy is still rising with λ, so the useful range is higher. Sweeping λ from 0.1 to 3.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef blk(F): F=F-F.mean(0); return F/np.sqrt((F**2).sum(1).mean())\ndef pcv(F, y, lams, seeds=(0,1,2)):\n    Y=np.eye(8)[y]; out={}\n    G=F@F.T\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                Ftm=F[tr].mean(0); Gc=(F-Ftm)@(F[tr]-Ftm).T\n                A=np.linalg.solve(Gc[tr]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=Gc[te]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\nZ=z(M[8]); Zn=Z/np.linalg.norm(Z,axis=1,keepdims=True)*np.sqrt(760)\nL2b=blk(z(M[2])); L6b=blk(z(M[6])); Lt=blk(Zn)\nlams=(0.1,0.3,1,3)\nprint('lin(8,2,6)', pcv(np.hstack([Lt,L2b,L6b]),y,lams))\nrng=np.random.RandomState(0)\nfor a in (0.5,1.0,2.0):\n    Wr=rng.randn(760,760)/np.sqrt(760); b=rng.randn(760)*1.0\n    P=blk(np.tanh(a*(Zn@Wr.T)+b))\n    print('rand a',a, pcv(np.hstack([P,Lt,L2b,L6b]),y,lams))\n    print('rand a',a,'x2', pcv(np.hstack([2*P,Lt,L2b,L6b]),y,lams))"
}
```


## reasoning summary

I'm working out the λ scale for the kernel ridge: the features have a much smaller norm than the normalized kernels used before, so the regularization needs to be larger. I'll sweep λ over a wider range, starting from 0.3 and going up to around 1.

I'm also weighing the λ sweep against normalizing each feature block to unit average squared norm, so every block gets comparable weight before applying the regularization.



## message

The λ grid was off for these feature scales. I'll normalize each feature block to unit mean squared norm, then sweep the block weights and λ properly.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef blk(F): F=F-F.mean(0); return F/np.sqrt((F**2).sum(1).mean())\ndef pcv(F, y, lams, seeds=(0,1,2)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                mu=F[tr].mean(0); Ft=F[tr]-mu\n                A=np.linalg.solve(Ft@Ft.T+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=(F[te]-mu)@(Ft.T@A)\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\nZ=z(M[8]); Zn=Z/np.linalg.norm(Z,axis=1,keepdims=True)*np.sqrt(760)\nL2b=blk(z(M[2])); L6b=blk(z(M[6])); Lt=blk(Zn)\nprint('lin(8,2,6)', pcv(np.hstack([Lt,L2b,L6b]),y,(0.003,0.01,0.03)))\nrng=np.random.RandomState(0)\nfor a in (0.5,1.0):\n    Wr=rng.randn(760,760)/np.sqrt(760); b=rng.randn(760)*1.0\n    P=blk(np.tanh(a*(Zn@Wr.T)+b))\n    for wP in (1.0,2.0):\n        print('rand a',a,'wP',wP, pcv(np.hstack([wP*P,L2b,L6b]),y,(0.003,0.01,0.03)))\n        print('   +lin top', pcv(np.hstack([wP*P,Lt,L2b,L6b]),y,(0.003,0.01,0.03)))"
}
```


## reasoning summary

I'm setting up a comparison of the base subset against the m=2816 and m=2560 (pass_5) variants, using the X1 formula features. I'll evaluate them with RBF leave-one-out, 5-fold cross-validation repeated over 5 seeds, and GELU kernel with both 5-fold and 20-fold.



## message

Now I'll compare base, m=2816 (pass_7) and m=2560 (pass_5) on identical 1024-id folds, using the exact pooling-layer features. Metrics are RBF LOO, RBF 5-fold×5 and GELU-kernel 5-fold×5/20-fold×2.

## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nimport json\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=1024; ys=y[:n]\nm7=json.load(open('pass_7/meta.json')); assert m7['ids']==[f'p{i:04d}' for i in range(n)]\nO=W('bert.encoder.layer.11.attention.output.dense.weight').astype(np.float64)\ng1=W('bert.encoder.layer.11.attention.output.LayerNorm.weight'); b1=W('bert.encoder.layer.11.attention.output.LayerNorm.bias')\ndef ln(X,g,b,eps=1e-12):\n    mu=X.mean(1,keepdims=True); v=((X-mu)**2).mean(1,keepdims=True); return (X-mu)/np.sqrt(v+eps)*g+b\ndef x1(C,M): return ln(C.astype(np.float64)+M.astype(np.float64)@O.T,g1,b1)[:,KEEP]\nfeats={'base':x1(d['cls_hidden_L9'][:n],d['mean_hidden_L9'][:n]),\n       'm2816':x1(np.load('pass_7/cls_hidden_L9.npy'),np.load('pass_7/mean_hidden_L9.npy')),\n       'm2560':x1(np.load('pass_5/cls_hidden_L9.npy'),np.load('pass_5/mean_hidden_L9.npy'))}\ndef cv(F,k,seeds,kind,lam):\n    accs=[]\n    for s in seeds:\n        P=np.zeros((n,8))\n        for te in folds(n,k,s,ys):\n            tr=np.setdiff1d(np.arange(n),te); Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n            T=np.eye(8)[ys[tr]]; T=T-T.mean(0)\n            if kind=='rbf': Ktr=rbf(Ztr,Ztr,0.4); Kte=rbf(Zte,Ztr,0.4)\n            else: Ktr=gelu(Ztr@Ztr.T/dd); Kte=gelu(Zte@Ztr.T/dd)\n            P[te]=Kte@np.linalg.solve(Ktr+lam*np.eye(len(tr)),T)\n        accs.append((P.argmax(1)==ys).mean())\n    return round(np.mean(accs),4)\nfor nm,F in feats.items():\n    print(nm, 'LOO rbf', round(krr_loo(F,ys,0.4,0.003),4), 'rbf5x5', cv(F,5,range(5),'rbf',0.003), 'gelu5x5', cv(F,5,range(5),'gelu',0.01), 'gelu20x2', cv(F,20,range(2),'gelu',0.01))\n"
}
```


## reasoning summary

The per-token scale looks fairly uniform, so reweighting tokens probably won't gain much. The variance of the second gain term swings quite a bit, though, so weighting selection by its square might help a little. I want to check a selection score on post-LayerNorm activations against the plain version, and also watch for outlier dimensions in DistilRoBERTa, which has some very large activation channels.

I'm also checking the outlier dimensions in DistilRoBERTa, since the huge activations in a few channels mean the LayerNorm error there gets scaled by the gain, so I'll run the cross-validation on that.



Per-token σ is nearly uniform, so token weighting is likely to matter little; I'll test only the gain-weighted unit selection with grouped cross-validation in post-LN2 space.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom lsbuild import gelu, layer_data\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\ndef cv_layer(l, m, metric='plain', lam=100.0, wcls=20.0, weos=5.0, nf=4):\n    a,W1,b1,U,b2=layer_data(l,base)\n    a2=np.load(f'true2_a_L{l}.npy').astype(np.float64)\n    tk=np.concatenate([np.load('tok_kind.npy'),np.load('tok2_kind.npy')])\n    grp=np.concatenate([np.load('tok_grp.npy'),1000+np.load('tok2_grp.npy')])\n    A=np.vstack([a,a2]); G=gelu(A@W1.T+b1); X=A+G@U.T+b2\n    pre=f'roberta.encoder.layer.{l}.'\n    g2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\n    H=LN(X,g2,bb)\n    w=np.where(tk==0,wcls,np.where(tk==1,weos,1.0))\n    ug=np.unique(grp); rng=np.random.RandomState(0); fold_of={g:i%nf for i,g in enumerate(rng.permutation(ug))}\n    fo=np.array([fold_of[g] for g in grp])\n    errs=[];errsF=[]\n    for f in range(nf):\n        tr=fo!=f; te=(fo==f)&(tk==2)\n        Gt=G[tr]; wt=w[tr]; sw=np.sqrt(wt)[:,None]\n        mu=(wt[:,None]*Gt).sum(0)/wt.sum(); Gc=(Gt-mu)*sw\n        if metric=='plain': sc=(U**2).sum(0)*(Gc**2).sum(0)\n        else: sc=((g2[:,None]*U)**2).sum(0)*(Gc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n        AA=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n        D=np.linalg.solve(AA+lam*np.eye(m),B).T\n        Un=U.copy(); Un[:,keep]+=D; Un[:,pr]=0\n        b2n=b2+U[:,pr]@mu[pr]-D@mu[keep]\n        Xe=A[te]+G[te]@Un.T+b2n\n        He=LN(Xe,g2,bb)\n        Ht=H[te]; errs.append(((He-Ht)**2).sum()/((Ht-Ht.mean(0))**2).sum())\n        Ft=X[te]-A[te]; errsF.append(((Xe-X[te])**2).sum()/((Ft-Ft.mean(0))**2).sum())\n    return np.mean(errs), np.mean(errsF)\nfor l in [0,1,2]:\n    for m in [1536,1792]:\n        for met in ['plain','g2']:\n            e,eF=cv_layer(l,m,met); print(l,m,met,'postLN relerr %.4f  ffn relerr %.4f'%(e,eF))\n"
}
```


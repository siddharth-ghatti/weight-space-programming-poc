## reasoning summary

I'm checking that U works out to roughly cos, since the standardized Z norm is about sqrt(768). Then I'll run a=2, t=0, lam=1, which gives 98.0%, and plan to sweep a, t and lam over a few values.

I'm also planning to sweep the lambda values over 1, 3 and 10, plus rho.



## message

The kernel features already reach ~98.0% CV (a=2, t=0, λ=1) vs 96.9% linear. Next I'll refine the grid over scale a, offset t, λ and the weight ρ of the linear part.

## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy'); E=np.load('feats/edA_cls12.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ncache={}\ndef feats(f,a,t):\n    tr=folds!=f\n    mu=E[tr].mean(0); sd=E[tr].std(0)+1e-6\n    Z=(E-mu)/sd; R=np.sqrt(768)\n    C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True)\n    return gelu(a*(Z@C.T)/R+t), Z/R\ndef run(a,t,lams,rhos):\n    out={}\n    FS=[feats(f,a,t) for f in range(5)]\n    for lam in lams:\n        for rho in rhos:\n            pr=np.zeros(len(lab),int)\n            for f in range(5):\n                tr=folds!=f; te=~tr\n                F=np.concatenate([FS[f][0], rho*FS[f][1]],1) if rho>0 else FS[f][0]\n                Ftr=F[tr]; m=Ftr.mean(0); Y=np.eye(8)[lab[tr]]-1/8\n                Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n                pr[te]=((F[te]-m)@Wt).argmax(1)\n            out[(lam,rho)]=round((pr==lab).mean(),4)\n    return out\nfor a,t in [(1,0),(2,0),(2,0.5),(3,0),(3,0.5),(1.5,0)]:\n    r=run(a,t,[0.3,1,3,10],[0,1,3])\n    best=max(r.items(),key=lambda kv:kv[1]); print(a,t,'best',best, r)\n"
}
```


## reasoning summary

Checking the tanh tail: at (0.5,-3) it gets 96.59%, the same as the exp Nyström run. Full KRR sits at 97.5%, so the gap likely comes from using plain ridge on features, which penalizes ||w||² rather than KRR's α^T K α. The proper Nyström KRR objective would be minimizing ||Y - F w||² + λ...

The regularized least-squares form needs the λK_AA term, so I'm switching to a generalized ridge solve: w = (FᵀF + λK_AA)⁻¹FᵀY. That's the closed-form version I can compute directly, and it should line up with the subset-of-regressors KRR result.



The tanh tail at (0.5, −3) matches the exp kernel at 96.59%, so the remaining gap is from using plain ridge instead of kernel ridge. I'm trying the subset-of-regressors form next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef sor_cv(X, y, fn, k=10, seeds=(0,), lams=(0.01,0.1,1), nanc=768, jit=1e-6):\n    res={lam:[] for lam in lams}\n    for s in seeds:\n        f=folds(y,k,s); S={lam:np.zeros((len(y),8)) for lam in lams}\n        for i in range(k):\n            tr=f!=i; te=f==i\n            mu=X[tr].mean(0); Z=X-mu; Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)\n            rng=np.random.RandomState(i+100*s); ai=rng.choice(tr.sum(),nanc,replace=False); A=Z[tr][ai]\n            F=fn(Z@A.T); Kaa=fn(A@A.T)\n            Ftr=F[tr]; fm=Ftr.mean(0); Fc=Ftr-fm; Y=np.eye(8)[y[tr]]; ym=Y.mean(0)\n            for lam in lams:\n                w=np.linalg.solve(Fc.T@Fc+lam*Kaa+jit*np.eye(nanc), Fc.T@(Y-ym))\n                S[lam][te]=(F[te]-fm)@w+ym\n        for lam in lams: res[lam].append((S[lam].argmax(1)==y).mean())\n    return {lam:round(float(np.mean(v)),4) for lam,v in res.items()}\nX=M(0)\nL=(0.003,0.01,0.03,0.1)\nprint('exp sor768', sor_cv(X,y,lambda G: np.exp(G),lams=L))\nprint('tanh(0.5,-3) sor768', sor_cv(X,y,lambda G: np.tanh(0.5*G-3)+1,lams=(1e-5,1e-4,1e-3,1e-2)))"
}
```


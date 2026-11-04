## reasoning summary

I'm checking how much of the FFN output's variance goes unexplained at each width: about 12% at m=1024, 6.5% at m=1536, and 3% at m=2048. Since the FFN output norm is smaller than X, the error looks smaller relative to the pre-LN sum, and I'm working through that fraction. I also want to try a higher λ and a better selection method, such as a greedy one.

I'm now weighing a backward-elimination selection that removes units by an Optimal Brain Surgeon-style cost, batching the removals since a full inverse update per unit would be too slow.

I'm weighing whether the iterative elimination counts as iterative fitting, and I think it doesn't, since it only chooses units while each weight fit stays closed-form. To stay clearly on the safe side, I'll use a one-shot criterion instead: an OBS-style cost computed once from the full ridge solution, and then I'll compare that against the energy-based selection.



Held-out error for layer 0 is 22% at m=512, 12% at m=1024, 6.5% at m=1536 and 3% at m=2048. Next I'll try an OBS-style one-shot selection with stronger ridge, and measure error against the full pre-LayerNorm sum.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\nimport scipy.special as sp, os\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\nZp=np.load('pass_10/token_hidden_L1.npy'); msk=np.load('pass_10/attention_mask.npy').astype(bool)\nZ=Zp[msk].astype(np.float64); sent=np.repeat(np.arange(64)[:,None],Zp.shape[1],1)[msk]\nP='bert.encoder.layer.0.'\nX=Z*base(P+'attention.output.LayerNorm.weight')+base(P+'attention.output.LayerNorm.bias')\nW1=base(P+'intermediate.dense.weight').astype(np.float64); bb1=base(P+'intermediate.dense.bias')\nW2=base(P+'output.dense.weight').astype(np.float64); bb2=base(P+'output.dense.bias')\nA=gelu(X@W1.T+bb1); F=A@W2.T+bb2\ntr=sent%2==0; te=~tr\nS=X+F; Sden=np.linalg.norm(S[te]-S[te].mean(0))**2\ndef fit_eval(keep,lam):\n    As=A[:,keep]; mu=As[tr].mean(0); Fm=F[tr].mean(0)\n    Ac=As[tr]-mu; Fc=F[tr]-Fm; W0=W2[:,keep].T\n    Wn=np.linalg.solve(Ac.T@Ac+lam*np.eye(len(keep)), Ac.T@Fc+lam*W0); b=Fm-mu@Wn\n    pred=As@Wn+b; e=np.linalg.norm(pred[te]-F[te])**2\n    return e/np.linalg.norm(F[te]-F[te].mean(0))**2, e/Sden\nAc=A[tr]-A[tr].mean(0)\nG=Ac.T@Ac\nen=A[tr].var(0)*(W2**2).sum(0)\nfor lam in [10,100]:\n    Ginv=np.linalg.inv(G+lam*np.eye(3072))\n    obs=(W2**2).sum(0)/np.diag(Ginv)\n    for m in [768,1024,1536]:\n        ko=np.sort(np.argsort(-obs)[:m]); ke=np.sort(np.argsort(-en)[:m])\n        print('lamsel',lam,'m',m,'OBS',['%.3f/%.4f'%fit_eval(ko,l) for l in [30,100,300]],' energy',['%.3f/%.4f'%fit_eval(ke,l) for l in [30,100,300]])\n"
}
```


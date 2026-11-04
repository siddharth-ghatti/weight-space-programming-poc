## reasoning summary

The model's numpy kernel classifier matches exactly, with a max logit difference of 6e-4. Next I'm weighing improvements, starting with better base features: concatenating mean activations across layers lifted linear accuracy to 98.1% from 97.2%, so a kernel on those features might reach around 98.5%. The catch is that only the mean of layer 11 reaches the final CLS token, which makes that harder to realize, though I'm starting to think it may not be as hard as it first looked.

I'm testing whether the final pooling could be fed by layer 10 as well: keep CLS's layer-10 output carrying the mean of that layer, then let the layer-11 residual mix it with a scaled mean of layer 11 so both signals survive. The worry is whether changing CLS at layer 10 would leak into the other tokens, and I'm checking that the answer is no.

I'm considering whether a layer-10 change could be CLS-specific, since modifying attention there would ripple into every token's output. One option is to repurpose a head to uniform attention so all tokens shift equally, then refit the kernel on the new features. A second idea is to use the fact that CLS states in late BERT layers are distinctive, so a CLS-only query might isolate the change.

I'm checking whether a CLS-only change at layer 10 is possible. Since the query is linear, I can't zero it out for just the CLS token, and shifting keys with a constant only cancels out rather than flattening attention, so I'm looking at whether layer norm might offer a way to shrink CLS's query selectively.

I'm thinking about whether a kernel on the concatenated means of several layers would do better than the mean of layer 11 alone, since that's where the errors seem to sit. I'll check cross-validation on those multi-layer features first: if they reach 98.5% or more, the extra work is worth it, and if not, I'll drop the idea.



The real model matches the closed-form kernel classifier exactly, with 100% argmax agreement. Before refitting on all centres, I'll test whether concatenated multi-layer features raise the ceiling.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef lnorm(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ndef run(E,fo,a,t,lams,rho=1.0):\n    out={}; FS=[]; R=np.sqrt(E.shape[1])\n    for f in range(5):\n        tr=fo!=f; mu=E[tr].mean(0); sd=E[tr].std(0)+1e-6\n        Z=(E-mu)/sd; C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True)\n        FS.append(np.concatenate([gelu(a*(Z@C.T)/R+t), rho*Z/R],1))\n    for lam in lams:\n        pr=np.zeros(len(lab),int)\n        for f in range(5):\n            tr=fo!=f; te=~tr; F=FS[f]; Ftr=F[tr]; m=Ftr.mean(0); Y=np.eye(8)[lab[tr]]-1/8\n            Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n            pr[te]=((F[te]-m)@Wt).argmax(1)\n        out[lam]=round((pr==lab).mean(),4)\n    return out\nM={l:lnorm(np.load(f'feats/mean_L{l}.npy')) for l in range(13)}\nfor name,ls in [('11',[11]),('10+11',[10,11]),('8+11',[8,11]),('6,8,10,11,12',[6,8,10,11,12]),('0+11',[0,11]),('12',[12]),('9',[9])]:\n    E=np.concatenate([M[l] for l in ls],1)\n    print(name, run(E,folds,2,0,[0.3,1]))\n"
}
```


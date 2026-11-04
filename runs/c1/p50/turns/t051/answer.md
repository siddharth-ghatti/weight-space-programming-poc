## reasoning summary

I'm working through which feature combinations might reach the higher accuracy, noting that errors are semantic and contextual models could help with some of them. I'm weighing whether kernels over block-normalized concatenations of the base features, possibly with CLS features, could close the gap, since the deeper-layer mixtures reportedly get to about 98.3%.

I'm settling on the block-normalized concatenation of M0, M2, M6, and M10, restricted to the KEEP dimensions, as the feature set for the kernel.



Bag-of-words misses the negation and framing errors, so I'm running kernel CV on the non-leaky base-layer mean features to see how much depth would add.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\ndef feat(l, kind='mean'):\n    X=B[f'{kind}_hidden_L{l}'][:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1,2), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return {g:max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,)).values()) for g in gs}\nfor combo in [[0,1],[0,2],[0,4],[0,6],[0,12],[0,2,6],[0,2,6,10],[0,1,2,3]]:\n    Z=np.concatenate([feat(l) for l in combo],1)\n    print(combo, kcv(Z))\nZ=np.concatenate([feat(0),feat(12,'cls')],1); print('M0+C12', kcv(Z))\n"
}
```


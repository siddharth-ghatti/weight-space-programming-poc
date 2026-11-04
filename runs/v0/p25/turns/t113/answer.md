## reasoning summary

Beta 1.4 is breaking, most likely from ill-conditioning in the kernel matrix, so I'll stick with beta=1. That leaves lambda around 0.02 to 0.05.

I'm also wondering whether float32 is the culprit: G and K are float32, and solving in float32 with a small lambda could be the source of the instability.

I'm switching to float64 for the final build to rule out float32 precision in the solve. Then I'll check which errors remain with the best config, and I'll print the seed 0 errors before writing a third batch aimed at them.



Regularisation at 0.02–0.05 holds steady around 0.988, so I'll keep beta=1 and use float64 for the final build, since beta=1.4 breaks numerically. The remaining out-of-fold errors on practice are next.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nk=np.load('kernel_v3.npz'); f3=np.load('my3_feats.npz')\nC=k['C'].astype(np.float64); yc=k['yc']; n=1496\nCall=np.vstack([C,f3['Z'].astype(np.float64)]); yall=np.concatenate([yc,f3['y']]); src=np.concatenate([np.zeros(n),np.ones(len(yc)-n),2*np.ones(len(f3['y']))]).astype(int)\nK=gelu(Call@Call.T/768)\nerrs={}\nfor seed in range(3):\n    f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n    for j in range(5):\n        trp=np.where(f!=j)[0]; te=np.where(f==j)[0]; tr=np.concatenate([trp, np.where(src>0)[0]])\n        w=np.where(src[tr]==0,1.0,0.3); Tm=np.eye(8)[yall[tr]]; b=(w[:,None]*Tm).sum(0)/w.sum()\n        A=np.linalg.solve(K[np.ix_(tr,tr)]+0.03*np.diag(1/w),Tm-b); P[te]=K[np.ix_(te,tr)]@A+b\n    for i in np.where(P.argmax(1)!=yc[:n])[0]: errs[i]=errs.get(i,0)+1; \n    print(seed,(P.argmax(1)==yc[:n]).mean())\n    if seed==0: P0=P\npr=practice()\nfor i in sorted(errs,key=lambda i:-errs[i]): print(errs[i], f\"[{LABELS[yc[i]][:10]} -> {LABELS[P0[i].argmax()][:10]}] {pr[i]['text']}\")\n"
}
```


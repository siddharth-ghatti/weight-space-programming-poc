## reasoning summary

I'm looking at why the RES dimensions have a much larger magnitude than the KEEP ones, at about 3.4 against 0.69. The carry term varies a lot across samples, so it takes a large share of the layer norm, roughly 12% of the total squared norm.

I'm weighing a fix for the RES dimensions: scale down their coefficient (c_r around 2) so they stop dominating the layer norm and making the per-sample norms of the KEEP block vary. Then I'm turning to the Cholesky overflow, where a near-zero pivot seems to blow up the division, and wondering whether the GELU(cos) kernel's numerical rank is lower than 1008.

I'm reading through the Cholesky failure now: once d[j] drops below about 1e-12 the division blows up, which suggests the effective rank of the kernel is under 1008 in float64. Anchors picked after that point may be garbage, possibly NaN, so I need a stopping rule based on the max of d or a fallback for the remaining anchors. Separately, I'm planning a small sweep over lam (0.1, 0.3) and c0 (0.25, 0.1), since the current 98.26 sits just under the 98.4 baseline.

I'm working out the RES scaling: if the carry term gets a smaller coefficient, the RES block shrinks toward the magnitude of C9_RES alone, which would take the 12% share of the layer norm away. I'm checking the rough size of the scale factors to see whether c_r around 2 would actually balance the two blocks.

I'm now checking the RES scaling by computing the z_RES magnitudes for c_r at 0, 1, and 2, since the linear block has only 8 features and its magnitude mostly doesn't matter.

I'm checking the RES scaling by computing z_RES magnitudes for a few coefficient values, and I want the cross-validation rerun with the more robust pivoted Cholesky. I'll also compare against an exact KRR with an exponential kernel on z, to see what the GELU(cos) features cost in accuracy.



The RES dims carry about 3.4 against 0.69 elsewhere under c=20, which skews the cosine features, so I'll give them a smaller gain, and pivoted Cholesky breaks down near 1,000 pivots, so it needs a stopping tolerance. After both fixes I'll retune λ/c0 and compare against exact KRR.



## tool_use: run_python

```json
{
 "code": "code=open('poolfit.py').read()\ncode=code.replace('''def pool_z(M9, C9, c, mu9):\n    return LN(c*(M9-mu9)+C9)''','''def pool_z(M9, C9, c, mu9, cr=1.0):\n    x=c*(M9-mu9)+C9\n    x[:,RES]=cr*(M9[:,RES]-mu9[RES])+C9[:,RES]\n    return LN(x)''')\ncode=code.replace('''def pivchol(K, k):\n    n=K.shape[0]; d=np.diag(K).copy(); L=np.zeros((n,k)); piv=[]\n    for i in range(k):\n        j=int(np.argmax(d)); piv.append(j)''','''def pivchol(K, k, tol=1e-8):\n    n=K.shape[0]; d=np.diag(K).copy(); L=np.zeros((n,k)); piv=[]\n    for i in range(k):\n        j=int(np.argmax(d))\n        if d[j]<tol*np.diag(K).max():\n            rest=[t for t in np.argsort(-np.diag(K)) if t not in set(piv)]\n            piv+= rest[:k-len(piv)]; break\n        piv.append(j)''')\nopen('poolfit.py','w').write(code)\nimport sys; sys.path.insert(0,'.')\nfrom poolfit import *\nfrom lib import labels, folds, krr_cv\ny=labels(); Y=np.eye(8)[y]\nF=np.load('feats_seq1024.npz'); M9=F['mean_hidden_L9'].astype(np.float64); C9=F['cls_hidden_L9'].astype(np.float64); M0=F['mean_hidden_L0'].astype(np.float64)\nmu9=M9.mean(0)\nfor cr in [0.0,1.0]:\n    Z=pool_z(M9,C9,20.0,mu9,cr); print('cr',cr,'zRES mag', np.abs(Z[:,RES]).mean(), 'KEEP', np.abs(Z[:,KEEP]).mean())\nZ=pool_z(M9,C9,20.0,mu9,1.0)\nL0=M0-M0.mean(0); L0/=np.sqrt((L0**2).sum(1).mean())\n# exact kernel reference on this z\nZk=Z[:,KEEP]; Zn=Zk/np.linalg.norm(Zk,axis=1,keepdims=True)\nKex=np.exp(Zn@Zn.T-1); Kg=gelu(Zn@Zn.T)\nprint('exactRBF+lin', krr_cv(Kex+0.5*L0@L0.T,y,[0.01],k=10,seeds=range(2)))\nprint('exactGELUcos+lin', {c0:krr_cv(Kg+c0*L0@L0.T,y,[0.003,0.01,0.03],k=10,seeds=range(2)) for c0 in [0.25,0.5,1.0]})\nev=np.linalg.eigvalsh(Kg); print('Kg eig min/max', ev.min(), ev.max(), (ev>1e-8*ev.max()).sum())\n"
}
```


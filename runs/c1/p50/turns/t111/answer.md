## reasoning summary

Checking the anchor count now: with every training point as an anchor (1346 in the 10-fold setup, which is all of them under the 1400 cap), the Nyström approximation is exact and scores 98.51 with the linear term. Dropping to 1000 anchors costs about 0.4%, landing around 97.86 without the linear term and 98.11 with it at λ=0.03. In the final model on 1496 sentences, m=1024 would mean roughly 1000 anchors, about 67% of the data, so I'm weighing whether that loss is acceptable.

I'm checking whether the pooler's 768 tanh units could serve as extra anchors to recover some of the accuracy lost from capping at 1000 anchors, since the final model has more training data than the CV setup. I'm also weighing better anchor selection methods like k-means or pivoted Cholesky to get more out of the same budget.

I'm considering whether the pooler's tanh units could add more anchors alongside the FFN units, so that the reserved dims get decoded and the total anchor count exceeds the 1496 training points. The tanh features would need some nonlinearity to act like a kernel, and I'm not yet sure a sigmoid-style kernel fits well with the Nyström setup.

I'm weighing a joint fit where the FFN's kernel scores land in the reserved dims and the pooler's tanh units add extra features, with the classifier weights combining both. The catch is that the FFN output gets normalized, so I need to check how that layer norm affects the reserved-dim scores.

I'm working out how the pooler's tanh units could decode the reserved dimensions, since the content part of the vector stays roughly fixed after layer norm if the reserved scores are kept small. That would let the decoding unit read each class score back out for the classifier.

I'm weighing whether to test anchor selection at 1000 and 750 anchors, comparing random picks against k-means centroids or greedy pivoted Cholesky with class balance. I'm also considering m=1536, which would let all 1496 practice points serve as anchors with 16 linear units and 24 spare, making the final model an exact kernel ridge fit at roughly 71.7M parameters.

I'm weighing m=1536, where all 1496 practice points serve as anchors and the final model becomes an exact kernel ridge fit at about 71.7M parameters. Against m=1024 at 63.8M, the pruned features and the anchor count both favor the larger model, so I'm checking whether the accuracy gain justifies the extra size. Since the score depends on parameter count among models that pass, I want to see the expected CV accuracy for the m=1536 path before deciding.

Weighing whether smarter anchor selection could let m=1024 get close to the all-anchor result, since a cheap offline test could settle it before committing to the larger m=1536 model.

Weighing whether smarter anchor selection could let m=1024 approach the all-anchor result, since a cheap offline test might settle it before committing to m=1536.



With every training point as an anchor, the penalized model matches exact KRR at 98.5%, and 1000 random anchors cost about 0.4%, which matters because m=1536 fits all 1496 practice sentences as anchors. Next I'll test whether pivoted-Cholesky or k-means anchors close that gap at 1000 and 750.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef cen(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\nZ9=cn(d['mean_hidden_L9']); C0=cen(d['mean_hidden_L0'])\ndef pivchol(K, m):\n    n=K.shape[0]; dg=np.diag(K).copy(); L=np.zeros((n,m)); sel=[]\n    for j in range(m):\n        i=int(np.argmax(dg)); sel.append(i)\n        l=(K[:,i]-L[:,:j]@L[i,:j])/np.sqrt(max(dg[i],1e-12)); L[:,j]=l; dg-=l**2; dg[i]=-1\n    return np.array(sel)\ndef kmeans_anchors(Zt, m, it=15, seed=0):\n    rng=np.random.RandomState(seed); Cc=Zt[rng.choice(len(Zt),m,replace=False)].copy()\n    for _ in range(it):\n        a=np.argmax(Zt@Cc.T,1)\n        for j in range(m):\n            w=Zt[a==j]\n            if len(w): Cc[j]=w.mean(0)\n        Cc/=np.linalg.norm(Cc,axis=1,keepdims=True)\n    return Cc\ndef nys_cv(sel='rand',nanch=1000,lam=0.01,c0=0.5,seeds=(0,1,2),k=10,Z=Z9,C=C0,yv=y):\n    accs=[]; n=len(yv); Yv=np.eye(8)[yv]\n    for s in seeds:\n        f=folds(yv,k,s); S=np.zeros((n,8)); rng=np.random.RandomState(100+s)\n        for i in range(k):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            if sel=='rand': A=Z[rng.choice(tr,nanch,replace=False)]\n            elif sel=='piv': A=Z[tr[pivchol(gelu(Z[tr]@Z[tr].T),nanch)]]\n            elif sel=='km': A=kmeans_anchors(Z[tr],nanch,seed=s)\n            Phi=gelu(Z@A.T); Kaa=gelu(A@A.T); na=len(A)\n            F=np.hstack([Phi, C]); Pen=np.zeros((na+C.shape[1],)*2); Pen[:na,:na]=Kaa+1e-6*np.eye(na); Pen[na:,na:]=np.eye(C.shape[1])/c0\n            mu=F[tr].mean(0); Fc=F-mu\n            Wt=np.linalg.solve(Fc[tr].T@Fc[tr]+lam*Pen, Fc[tr].T@(Yv[tr]-Yv[tr].mean(0)))\n            S[te]=Fc[te]@Wt\n        accs.append((S.argmax(1)==yv).mean())\n    return round(float(np.mean(accs)),4)\nfor nanch in [1000,750]:\n    for sel in ['rand','piv','km']:\n        print(nanch, sel, [nys_cv(sel=sel,nanch=nanch,lam=l) for l in [0.003,0.01,0.03]])"
}
```


## reasoning summary

The 98.4 and 98.53 scores at 10 and 20 folds are close to the base model's 98.46, so the pruned m=1024 sequential holds up well. I'm wondering whether pruning further to m=768 is worth it, but that would take another nine exposure passes, leaving only about 12 to 13 of my remaining 15 steps, which is tight but might fit. Before deciding, I want to finish the

I still need to finish the pool layer and check that the whole pipeline runs end to end, which takes one or two passes. After that I'll decide between pruning to m=768 and dropping a layer, since reading the cheaper M8 option now would have been better.

I'm weighing a cheaper cut against the 750-anchor question. Dropping layer 8 looks like it would cost roughly 0.2 points, versus about 9 more exposure passes for m=768. Reading the layer-8 hidden states is only two passes and I can evaluate it before committing, so I'll check that first.

I'm going to build the pool layer on the current 10-layer model first so there's a valid submission, and I'll read layer 8 in that same verification pass. That gives me both the check and the M8 features for a possible reduction.

I'm setting up the anchor-based feature map now: normalizing the pooled vectors, picking about 1008 anchors from the 1496 training points via pivoted Cholesky, and applying a GELU to the cosine similarities against those anchors.

I'm working out the penalty term for the anchor model, with the kernel matrix among anchors needing a jitter in case it isn't positive semidefinite. Next I want an honest cross-validation estimate, using a linear block scaled per sample and anchors picked only within each training fold, to compare against the 98.4 from the 9-layer version. I still need to check the scale of the residual features before fixing those constants.

I'm checking how the residual carry feeds into the normalized features, since those eight dimensions are a small part of the 768-dimensional vector. I'm fitting the anchor features together with the linear block, with a penalty applied to each part of the solve.

I'm writing the fit and cross-validation code for the anchor model, with anchors picked from training data inside each fold. I'm also checking whether the anchor kernel matrix is positive semidefinite before trusting the solve.

Writing the pivoted Cholesky and feature-building code for the pool layer now...

Still writing the pivoted Cholesky and fit code...



The sequential m1024 network scores 98.4% at 10-fold and 98.53% at 20-fold, close to the base model, so the pruning costs little. Next I'll build the numpy pool fit and run cross-validation to pick the ridge penalty, scaling factor and anchor count.



## tool_use: run_python

```json
{
 "code": "code=r'''\nimport numpy as np, scipy.special as sp\nRES=np.array([83,88,222,254,444,495,506,601]); KEEP=np.setdiff1d(np.arange(768),RES)\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ndef LN(u):\n    m=u.mean(-1,keepdims=True); v=((u-m)**2).mean(-1,keepdims=True); return (u-m)/np.sqrt(v+1e-12)\ndef pool_z(M9, C9, c, mu9):\n    return LN(c*(M9-mu9)+C9)\ndef anchor_dirs(Za):\n    A=Za.copy(); A[:,RES]=0\n    return A/np.linalg.norm(A,axis=1,keepdims=True)/np.sqrt(768)\ndef feats(Z, Wd): return gelu(Z@Wd.T)\ndef pivchol(K, k):\n    n=K.shape[0]; d=np.diag(K).copy(); L=np.zeros((n,k)); piv=[]\n    for i in range(k):\n        j=int(np.argmax(d)); piv.append(j)\n        l=(K[:,j]-L[:,:i]@L[j,:i])/np.sqrt(max(d[j],1e-12)); L[:,i]=l; d=d-l**2; d[piv]=-np.inf\n    return np.array(piv)\ndef fit(Phi, Lin, Y, Kaa, lam, c0, jitter=1e-6):\n    X=np.concatenate([Phi,Lin],1); mu=X.mean(0); Xc=X-X.mean(0); Ym=Y.mean(0); Yc=Y-Ym\n    na=Phi.shape[1]; nl=Lin.shape[1]\n    R=np.zeros((na+nl,na+nl)); R[:na,:na]=Kaa+jitter*np.trace(Kaa)/na*np.eye(na); R[na:,na:]=np.eye(nl)/c0\n    coef=np.linalg.solve(Xc.T@Xc+lam*R, Xc.T@Yc)\n    b=Ym-mu@coef\n    return coef, b\n'''\nopen('poolfit.py','w').write(code)\nimport sys; sys.path.insert(0,'.')\nfrom poolfit import *\nfrom lib import labels, folds\ny=labels(); Y=np.eye(8)[y]\nF=np.load('feats_seq1024.npz'); M9=F['mean_hidden_L9'].astype(np.float64); C9=F['cls_hidden_L9'].astype(np.float64); M0=F['mean_hidden_L0'].astype(np.float64)\nmu9=M9.mean(0)\nZ=pool_z(M9,C9,20.0,mu9)\nprint('zRES mag', np.abs(Z[:,RES]).mean(), 'KEEP', np.abs(Z[:,KEEP]).mean())\nL0=M0-M0.mean(0); L0/=np.sqrt((L0**2).sum(1).mean())\ndef cv(na, lam, c0, k=10, seeds=(0,1), lin=L0):\n    accs=[]\n    for s in seeds:\n        f=folds(y,k,s); S=np.zeros((len(y),8))\n        for i in range(k):\n            tr=f!=i; te=f==i\n            Ztr=Z[tr]; Kc=gelu(Ztr@anchor_dirs(Ztr).T)  # GELU(cos) among train\n            piv=pivchol(Kc, na); Wd=anchor_dirs(Ztr[piv])\n            Ptr=feats(Ztr,Wd); Kaa=Ptr[piv]\n            coef,b=fit(Ptr, lin[tr], Y[tr], (Kaa+Kaa.T)/2, lam, c0)\n            S[te]=np.concatenate([feats(Z[te],Wd),lin[te]],1)@coef+b\n        accs.append((S.argmax(1)==y).mean())\n    return round(float(np.mean(accs)),4)\nfor na in [1008]:\n    for lam in [0.01,0.03]:\n        for c0 in [0.5,2.0]:\n            print(na,lam,c0, cv(na,lam,c0))\n"
}
```


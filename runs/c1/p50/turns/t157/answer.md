## reasoning summary

I'm weighing whether the kernel's indefiniteness is actually a problem here: the exact dual solve needs a PSD matrix, which is why a=1, b=0 was picked before. With the anchor approach the features are just Phi = GELU(a S_za + b), so a plain ridge penalty on alpha would avoid needing PSD altogether, and the primal solve on the 1496×1008 features should work either way. The f

I'm planning a primal test: GELU features from anchors chosen by pivoted Cholesky on the exp kernel, with a ridge penalty on the output weights, checked by cross-validation inside each fold. The baseline to beat is the GELU-of-cosine Nyström result of 98.26, which was computed on contaminated z. Since the FFN unit weights can be arbitrary, I'll also consider that anchors are just one way to pick them, while still needing a closed-form rule because there is no training.



Indefinite kernels break exact dual KRR, so I'd fixed a=1, b=0 there, but plain ridge in the primal anchor model doesn't need a PSD kernel. Next I'll run CV over (a, b) and λ, picking anchors per fold via pivoted Cholesky on the exp kernel.



## tool_use: run_python

```json
{
 "code": "import sys, time; sys.path.insert(0,'.')\nfrom poolfit import *\nfrom lib import labels, folds\ny=labels(); Y=np.eye(8)[y]\nF=np.load('feats_seq1024.npz'); M9=F['mean_hidden_L9'].astype(np.float64); C9=F['cls_hidden_L9'].astype(np.float64); M0=F['mean_hidden_L0'].astype(np.float64)\nmu9=M9.mean(0); Z=pool_z(M9,C9,20.0,mu9,1.0)\nL0=M0-M0.mean(0); L0/=np.sqrt((L0**2).sum(1).mean())\nZk=Z[:,KEEP]; Zn=Zk/np.linalg.norm(Zk,axis=1,keepdims=True); S=Zn@Zn.T\nKex=np.exp(S-1)\ndef ridge(X, Y, pen):\n    mu=X.mean(0); Xc=X-mu; Ym=Y.mean(0)\n    coef=np.linalg.solve(Xc.T@Xc+np.diag(pen), Xc.T@(Y-Ym)); return coef, Ym-mu@coef\nt=time.time()\nf=folds(y,10,0); pivs=[]\nfor i in range(10):\n    tr=np.where(f!=i)[0]; pivs.append(tr[pivchol(Kex[np.ix_(tr,tr)],1008)])\nprint('piv time', time.time()-t)\nnp.save('cv_pivs_f10s0.npy', np.array(pivs))\ncfgs=[(1,0),(2,-0.5),(3,-1.5),(4,-2.5),(6,-4)]\nfor a,b in cfgs:\n    out={}\n    for lam in [0.01,0.1,1.0]:\n        for c0 in [0.5]:\n            Sc=np.zeros((1496,8))\n            for i in range(10):\n                tr=f!=i; te=f==i; P=gelu(a*S[:,pivs[i]]+b)\n                X=np.concatenate([P,L0],1); pen=np.r_[lam*np.ones(1008), lam/c0*np.ones(768)]\n                coef,b0=ridge(X[tr],Y[tr],pen); Sc[te]=X[te]@coef+b0\n            out[(lam,c0)]=round(float((Sc.argmax(1)==y).mean()),4)\n    print(a,b,out)\n"
}
```


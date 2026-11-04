## reasoning summary

I'm weighing whether the kernel's edge over random tanh features comes from the linear part or from normalization, since an RBF on z-scored data with a small lambda behaves a bit like nearest-neighbor. I want to see what kernel ridge with gamma 0.3 is actually picking up.

I'm testing whether a linear model can match the kernel: shrinkage LDA, and ridge on PCA-whitened top components. If a linear readout reaches around 97%, the nonlinear readout would be unnecessary.



Random tanh features top out at 94.5%, and centred ones don't help. Next I'll test whether the kernel's edge over ridge is nonlinearity or just better conditioning, using whitened linear methods.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\ndef lda_cv(X, shr_list, seeds=(0,1), whiten_only=False):\n    out={s_:[] for s_ in shr_list}\n    for s in seeds:\n        f=folds(y,5,s); S={s_:np.zeros((len(y),8)) for s_ in shr_list}\n        for i in range(5):\n            tr=f!=i; te=f==i\n            mu=X[tr].mean(0)\n            means=np.stack([X[tr][y[tr]==c].mean(0) for c in range(8)])\n            Xc=X[tr]-means[y[tr]]\n            Sw=Xc.T@Xc/len(Xc)\n            for s_ in shr_list:\n                Sig=(1-s_)*Sw+s_*np.trace(Sw)/Sw.shape[0]*np.eye(Sw.shape[0])\n                Wl=np.linalg.solve(Sig, (means-mu).T)\n                b=-0.5*np.sum((means-mu).T*Wl,0)\n                S[s_][te]=(X[te]-mu)@Wl+b\n        for s_ in shr_list: out[s_].append((S[s_].argmax(1)==y).mean())\n    return {k:round(float(np.mean(v)),4) for k,v in out.items()}\nfor L in [0,1,2,9,12]:\n    print(L, lda_cv(M(L), [0.01,0.03,0.1,0.3]))\n"
}
```


## reasoning summary

I'm checking the range of the u values: they run from about -0.56 up to 2.82, though that top value is just an outlier matching itself. Off-diagonal test points only reach roughly 1 to 2, and the approximation error is around 1% relative to the largest exponential term. That might be acceptable, but KRR predictions depend on cancellations, so I'm considering weighting the approximation toward where most of the u values fall, roughly -0.4 to 0.9.

I'm considering fitting the kernel ridge regression directly with a GELU-based kernel, κ(u) = c0 + Σ c_r GELU(g_r u + b_r), so the model I evaluate in cross-validation is exactly the one I deploy and there's no approximation error to worry about. The kernel only needs to be useful, not equal to exp, though the kernel matrix may not be positive semi-definite, so I'll need to check that its eigenvalues stay above -λ. A simpler version uses a single unit per center, κ(u) = GELU(g u + b), which I'd evaluate with 20-fold CV against a two-unit combination.



A 2-GELU fit to exp(0.8u) is only about 1% accurate on the u range, so I'll instead use κ(u) = Σ c_r·GELU(g_r·u + b_r) directly as the KRR kernel, which removes the approximation gap. Next I'm testing 1-unit and 2-unit GELU kernels with 20-fold CV and checking for negative eigenvalues.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\ny=np.load('base_feats.npz')['y']; X1=np.load('X1_full.npy'); F=X1[:,KEEP]\npre=[]\nfor s in (0,1):\n    for te in folds(len(y),20,s,y):\n        tr=np.setdiff1d(np.arange(len(y)),te)\n        Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n        pre.append((s,tr,te,Ztr@Ztr.T/dd,Zte@Ztr.T/dd))\ndef ev(kfun, lams):\n    out={l:{} for l in lams}; mineig=1e9\n    for s,tr,te,Gtr,Gte in pre:\n        T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n        Ktr=kfun(Gtr); Kte=kfun(Gte)\n        w,V=np.linalg.eigh(Ktr); mineig=min(mineig,w.min()); VT=V.T@T\n        for l in lams: out[l].setdefault(s,np.zeros((len(y),8)))[te]=Kte@(V@(VT/(w+l)[:,None]))\n    return {l:round(np.mean([(v.argmax(1)==y).mean() for v in out[l].values()]),4) for l in lams}, round(mineig,5)\nL=[0.001,0.003,0.01,0.03]\nprint('exp0.8', ev(lambda G: np.exp(0.8*G), L))\nfor g,b in [(0.8,1),(0.5,0),(1,0),(1,1),(0.8,2),(1.5,1)]:\n    print('gelu',g,b, ev(lambda G: gelu(g*G+b), L))\nprint('2unit', ev(lambda G: 10.551*gelu(0.5*G)+43.182*gelu(0.5*G-2), L))\n"
}
```


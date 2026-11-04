## reasoning summary

I've got the variance split figured out: the top 300 components of the C9 deviations carry 97.7% of the variance, and 468 components of M9 carry 98.75%, so I'm settling on r_C = 300 and r_M = 468. Then I'm asking whether the discriminative signal, not just the variance, might sit in those low-variance directions, since truncating M11 to 384 dimensions only cost 0.4% earlier.

I'm weighing whether to emulate the setup offline, building x as a layer-normalized mix of C9 and the top-468 M9 directions, with the bottom-468 C9 directions acting as noise on the M9 channel. I'm picking a scaling factor around 1 to match their comparable variances, and I'm wondering whether standardizing the features inside the ridge fit would be enough to handle the scale mismatch.

I'm now working out the kernel as an explicit linear map: per-unit weights scaled by the standardized center, with a bias that absorbs the mean terms, so the GELU unit becomes a cosine-style similarity in standardized space.

I'm now checking whether the linear and kernel models do better on the emulated features than on the standardized [C9, M9] set, starting with a GELU-feature ridge in place of an RBF, since GELU units can't reproduce an exponential kernel exactly.



C9's variance is concentrated in its top 300 PCs and M9's in its top 468, so mapping M9's top 468 PCs into C9's bottom 468 directions should add them with little interference. Next I'll compute x = LN(C9 + κ·E·W·(M9−M̄9)) offline and compare linear ridge against ridge on GELU kernel features.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\nF=np.load('feats.npz'); y=F['y']\nC=F['C9']; M=F['M9']\nUc=np.linalg.svd(C-C.mean(0),full_matrices=False)[2]   # rows = PCs\nUm=np.linalg.svd(M-M.mean(0),full_matrices=False)[2]\nrM=468\nE=Uc[768-rM:].T      # 768 x rM (bottom PCs of C9)\nWm=Um[:rM]           # rM x 768\ndef LN(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ndef mkfolds(seed):\n    rng=np.random.RandomState(seed); f2=np.zeros(len(y),int)\n    for c in range(8):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f2[idx]=np.arange(len(idx))%5\n    return f2\nFS=[mkfolds(s) for s in [0,7,11]]\ndef evalx(X, lams_lin, gelu_params=None, lams_k=None, wlin=1.0):\n    res_lin=np.zeros(len(lams_lin)); res_k={}\n    for fo in FS:\n        for f in range(5):\n            tr=fo!=f; te=~tr\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd\n            Y=np.eye(8)[y[tr]]-1/8\n            K=Z@Z[tr].T; ev,U=np.linalg.eigh(K[tr]); UtY=U.T@Y\n            for i,l in enumerate(lams_lin):\n                a=U@(UtY/(ev+l)[:,None]); res_lin[i]+=(np.argmax(K[te]@a,1)==y[te]).sum()\n            if gelu_params:\n                S=Z@Z[tr].T/Z.shape[1]\n                for (beta,theta) in gelu_params:\n                    Phi=gelu(beta*S-theta); Phi=Phi-Phi[tr].mean(0)\n                    G=np.concatenate([wlin*Z/np.sqrt(Z.shape[1]),Phi],1)\n                    Kg=G@G[tr].T; ev2,U2=np.linalg.eigh(Kg[tr]); UtY2=U2.T@Y\n                    for l in lams_k:\n                        a=U2@(UtY2/(ev2+l)[:,None]); res_k[(beta,theta,l)]=res_k.get((beta,theta,l),0)+(np.argmax(Kg[te]@a,1)==y[te]).sum()\n    n=len(FS)*len(y)\n    return np.round(res_lin/n,4), {k:round(v/n,4) for k,v in res_k.items()}\nfor kap in [0.5,1.0,2.0]:\n    X=LN(C + kap*((M-M.mean(0))@Wm.T)@E.T)\n    rl,_=evalx(X,[200,300,500,800])\n    print('kappa',kap,'linear on x',rl)\nnp.save('design_E_Wm.npy',np.stack([E.T,Wm]))"
}
```


## reasoning summary

I'm comparing the RBF kernel results: 98.3% on M9 and 97.3% on M0, and a smaller lambda might help. Next I want to try RBF with a smaller lambda, cosine normalization, and other layers, with kernel ridge on each M_L. I'm also wondering whether the pooler's 768 tanh units, maybe with FFN units, could approximate the kernel, though random features at that size only approximate it imperfectly.

I'm considering a Nyström-style variant where the 768 pooler units use training points as centers, so each unit computes a sigmoid-kernel-like feature, with ridge regression on top. I'd test that architecture by cross-validation, and I'm also checking whether the pooled input can be made close to the mean-pooled representation by letting the attention output dominate the pooling layer's LayerNorm.

I'm weighing whether the pooler could approximate the kernel by using training points as centers, and I'd want to check that with cross-validation. Next I'll look at which layer depth works best with RBF kernel ridge, using a small grid over gamma and lambda, and also try the kernel on per-sample normalized means, since that is what the pooler actually sees.

I'm going to check the realizable version directly, using 768 tanh features built from random or training-point centers rather than the exact kernel.



The RBF kernel ridge lifts accuracy to 98.3% on the layer-9 mean and 97.3% on the embedding mean. Next I'm scanning every layer with that kernel for the shallowest one reaching 98%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef kfeat(X, tr):\n    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n    Z=(X-mu)/sd; return Z/np.sqrt(Z.shape[1])\ndef krr_cv(X, y, gamma, lams, k=5, seed=0):\n    n=len(y); S={l:np.zeros((n,8)) for l in lams}\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        Z=kfeat(X,tr); sq=(Z**2).sum(1)\n        K=np.exp(-gamma*np.maximum(sq[:,None]+sq[None,:]-2*Z@Z.T,0))\n        Ktr=K[np.ix_(tr,tr)]; T=np.eye(8)[y[tr]]; tm=T.mean(0)\n        w,V=np.linalg.eigh(Ktr); R=V.T@(T-tm)\n        for l in lams:\n            a=V@(R/(w+l)[:,None]); S[l][te]=K[np.ix_(te,tr)]@a+tm\n    return {l:round((S[l].argmax(1)==y).mean(),4) for l in lams}\nfor L in range(13):\n    out={g:krr_cv(M[L],y,g,(0.001,0.01,0.03)) for g in (0.2,0.4)}\n    print(L, out)\n"
}
```


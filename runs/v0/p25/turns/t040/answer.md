## reasoning summary

The network matches my designed function to about 1e-4, so the construction works. I estimate it's around 98 to 98.5 percent accurate, and I'm asking whether I can push that higher. One route is to enrich the features the kernel gets: right now they come from uniform attention with the value matrix set to identity, and the other heads could compute different weighted means to carry more information. Another is tuning the kernel itself, starting with how the features are weighted.

I'm weighing a few ways to improve the kernel's inputs and tuning, including LDA-projected features, which would need nested CV to avoid leakage. I'm also considering multi-scale neurons with two β values, since 1496 × 2 fits within the 3072 budget, and I plan to test these offline since the features are computable exactly.

I'm going to test a few supervised variants of the kernel's input space: the current standardized features, features whitened by within-class covariance with shrinkage, and features projected onto the top-k PCA components before standardizing. Any LDA-based version has to be fit inside each training fold so the CV stays clean.



The network matches the designed kernel machine to within 1e-4 on every logit, and v2 is in place at an honest CV estimate of about 98.0–98.5%. Next I'll look for further offline gains: PCA truncation of the kernel input, a within-class whitening metric fit inside each fold, and multi-scale neurons.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nfrom scipy.special import erf as serf\nK=np.load('kernel_v2.npz'); n=K['n']; Z=K['Z']\ny=np.load('base_feats.npz')['y']; folds=np.load('folds.npy')\nT=np.eye(8)[y]\ngelu=lambda u: 0.5*u*(1+serf(u/np.sqrt(2)))\ndef mkfolds(seed,k):\n    rng=np.random.default_rng(seed); f=np.zeros(len(y),int)\n    for c in range(8):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f[idx]=np.arange(len(idx))%k\n    return f\nFOLDS=[folds,mkfolds(11,5),mkfolds(12,10)]\ndef feat_cv_feats(featfn, lam, fo):\n    # featfn(tr) -> P (n x m) features computed using only training rows tr for any fitting\n    pred=np.zeros(len(y),int)\n    for f in range(fo.max()+1):\n        tr=fo!=f; P=featfn(tr); Ptr=P[tr]; pm=Ptr.mean(0); A=Ptr-pm\n        a=np.linalg.solve(A@A.T+lam*np.eye(len(A)),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax((P[~tr]-pm)@(A.T@a),1)\n    return (pred==y).mean()\ndef report(name,featfn,lams=[0.003,0.01,0.03]):\n    r=[[round(feat_cv_feats(featfn,l,fo),4) for l in lams] for fo in FOLDS]\n    print(name, r, 'avg best-lam', np.round(np.mean(r,0),4),flush=True)\nG=Z@Z.T/768\nreport('baseline b1',lambda tr: gelu(G[:,tr]))\nreport('multiscale .75+1.5',lambda tr: np.hstack([gelu(0.75*G[:,tr]),gelu(1.5*G[:,tr])]))\n# PCA truncation (unsupervised, all data)\nnc=n-n.mean(0); Uu,S,Vt=np.linalg.svd(nc,full_matrices=False)\nfor k in [128,256]:\n    Zk=nc@Vt[:k].T; Zk=Zk/Zk.std(0); Gk=Zk@Zk.T/k\n    report(f'pca{k} whiten',lambda tr,Gk=Gk: gelu(Gk[:,tr]))\n    Zk2=nc@Vt[:k].T; Gk2=Zk2@Zk2.T/ (Zk2**2).sum(1).mean()\n    report(f'pca{k} raw',lambda tr,Gk2=Gk2: gelu(Gk2[:,tr]))"
}
```


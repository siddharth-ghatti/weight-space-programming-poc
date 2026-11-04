## reasoning summary

I'm checking whether the basis regression behaves differently from kernel ridge, since the effective kernel here is Φ Φ^T. The GELU setup at λ=0.1 gave 0.981, so I want to see whether a smaller λ helps, and I'm scanning β over {1, 2, 3} along with some negative thresholds.

I'm also considering kernel ridge directly on a GELU kernel K_ij = GELU(β G_ij − θ), solving for coefficients α = (K + λI)^{-1} T, so the fitted function has the same neuron-centered form but is fit differently from the feature-based ridge. I'll test both approaches.



Neurons with GELU(2·cos-similarity) centered at training samples reach 98.1% under honest CV. Next I'll tune β, θ and λ, and compare kernel-ridge fitting against feature-ridge fitting for the same neuron form.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nfrom scipy.special import erf as serf\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nx=c11+2*m11; xc=x-x.mean(1,keepdims=True); n=xc/xc.std(1,keepdims=True)\nmu=n.mean(0); sd=n.std(0)+1e-6; Z=(n-mu)/sd; d=768\nT=np.eye(8)[y]\ngelu=lambda u: 0.5*u*(1+serf(u/np.sqrt(2)))\nG=Z@Z.T/d\ndef feat_cv(fn, lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; P=fn(G[:,tr]); Ptr=P[tr]; pm=Ptr.mean(0); A=Ptr-pm\n        a=np.linalg.solve(A@A.T+lam*np.eye(len(A)),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax((P[~tr]-pm)@(A.T@a),1)\n    return (pred==y).mean()\ndef krr_cv(fn, lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; K=fn(G[np.ix_(tr,tr)])\n        a=np.linalg.solve(K+lam*np.eye(len(K)),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax(fn(G[np.ix_(~tr,tr)])@a,1)\n    return (pred==y).mean()\nfor beta,theta in [(1,0),(1.5,0),(2,0),(3,0),(2,-1),(3,1)]:\n    fn=lambda g,b=beta,t=theta: gelu(b*g-t)\n    print('feat',beta,theta,[round(feat_cv(fn,l),4) for l in [0.01,0.03,0.1]],\n          '| krr',[round(krr_cv(fn,l),4) for l in [0.01,0.1,1]],flush=True)"
}
```


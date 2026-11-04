## reasoning summary

I'm checking a feature-ridge setup where β=1 with λ=0.01 gives about 0.985, and β=1.5 with λ=0.03 gives 0.984. Next I want to try smaller β values and smaller λ, since for small β the GELU behaves roughly like a linear plus quadratic term, which fits the idea of a quadratic-plus-linear kernel combination.

I'm weighing whether to pick a robust region with β between 1 and 1.5 and λ around 0.01 to 0.03, given the CV noise is roughly five samples. I'm also considering other feature vectors, like different mixing constants or the mean-only version, and planning a small scan at β=1 across a few λ values.



Thresholdless GELU neurons reach about 98.4–98.5% honest CV, up from 97.5% for v1. I'm scanning smaller β, λ and mixing weights c to find a stable optimum rather than a lucky point.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nfrom scipy.special import erf as serf\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nT=np.eye(8)[y]\ngelu=lambda u: 0.5*u*(1+serf(u/np.sqrt(2)))\ndef prep(c):\n    x=c11+c*m11; xc=x-x.mean(1,keepdims=True); n=xc/xc.std(1,keepdims=True)\n    Z=(n-n.mean(0))/(n.std(0)+1e-6); return Z@Z.T/768\ndef feat_cv(G,fn, lam, folds=folds):\n    pred=np.zeros(len(y),int)\n    for f in range(folds.max()+1):\n        tr=folds!=f; P=fn(G[:,tr]); Ptr=P[tr]; pm=Ptr.mean(0); A=Ptr-pm\n        a=np.linalg.solve(A@A.T+lam*np.eye(len(A)),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax((P[~tr]-pm)@(A.T@a),1)\n    return (pred==y).mean(), pred\nfor c in [1,2,4]:\n    G=prep(c)\n    for beta in [0.5,0.75,1.0,1.25]:\n        fn=lambda g,b=beta: gelu(b*g)\n        print('c',c,'beta',beta,[round(feat_cv(G,fn,l)[0],4) for l in [0.001,0.003,0.01,0.03]],flush=True)"
}
```


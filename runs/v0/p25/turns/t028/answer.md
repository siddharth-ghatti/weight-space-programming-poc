## reasoning summary

I'm weighing whether raw coordinates are enough here, and they aren't: the bow benefit needs the full mean-zero version or a supervised projection. A supervised projection from layer 0 gave 0.980 under nested CV, but building it with FFN pair neurons on the post-attention state raises a problem, since that input differs from x_j, and the final fit may leak.

I'm checking whether the per-token bow evidence, w·x_j, can be linearly read out of layer 11's states. If it could, linear mean pooling would already carry it and the combinations wouldn't add anything, but they do, so it isn't fully decodable. Layer 6's mean has a similar limitation, so I'm looking at nonlinear per-token features before pooling instead, including the layer 11 FFN applied to the pooled CLS vector.

I'm now sketching a nonlinear route: applying the layer 11 FFN to the pooled CLS vector, with random GELU features and a closed-form ridge readout on top. The catch is that the FFN output has to be written back into the 768-dim residual before the pooler and classifier, and I'd need the intermediate activations, which means reproducing a layer's forward pass in numpy.

I'm checking whether a nonlinear readout on the layer 11 features could beat the linear one, since the FFN route would mean reproducing a layer's forward pass. I'll test kernel ridge with an RBF kernel on the standardized mean-pooled features against the linear baseline, using cross-validation. I'm also considering better normalization or whitening of the pooled features, since the designed ridge concatenation may be leaving something on the table.

I'm going to try a few classifier variants on the cls11 and mean11 features together, mainly kernel methods, to see whether anything can get past 0.975 from those two alone.



Raw embedding coordinates give no gain on their own, so I'm checking offline whether a nonlinear readout (kernel ridge with RBF or polynomial kernels) beats a linear one on the final layer's cls11 and mean11 features.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nX=np.hstack([c11,m11]); X=(X-X.mean(0))/(X.std(0)+1e-6)\nT=np.eye(8)[y]\ndef krr_cv(K,lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f\n        a=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax(K[np.ix_(~tr,tr)]@a,1)\n    return (pred==y).mean()\nsq=(X**2).sum(1); D2=sq[:,None]+sq[None,:]-2*X@X.T\nmed=np.median(D2)\nfor gmul in [0.25,0.5,1,2]:\n    K=np.exp(-D2/(gmul*med))\n    print('rbf',gmul,[round(krr_cv(K,l),4) for l in [0.01,0.1,1]],flush=True)\nKl=X@X.T\nprint('linear kernel',[round(krr_cv(Kl,l),4) for l in [100,300,1000]])"
}
```


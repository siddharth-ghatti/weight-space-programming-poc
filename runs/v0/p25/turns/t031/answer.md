## reasoning summary

The random tanh features are much worse here, probably because 768 features on 768-dimensional standardized input leaves the ridge penalty too weak; λ up to 10 looks too small, so I'll try 100 and 1000. Small σ works best, which pushes toward the near-linear regime, and as σ goes to zero it reduces to plain linear ridge at 0.97, so I'm testing larger λ next.

I'll sweep a grid of λ values from 30 to 1000 alongside σ values from 0.25 to 1.



## message

Random tanh features underperform here, but the ridge penalty range was clearly too small (768 features vs ~1200 training samples). Let me scan much larger λ.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nx=c11+2*m11; xc=x-x.mean(1,keepdims=True); n=xc/xc.std(1,keepdims=True)\nmu=n.mean(0); sd=n.std(0)+1e-6; Z=(n-mu)/sd\nT=np.eye(8)[y]\ndef ridge_plain(F,lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; Fm=F[tr].mean(0); A=F[tr]-Fm\n        Wt=np.linalg.solve(A.T@A+lam*np.eye(F.shape[1]),A.T@(T[tr]-T[tr].mean(0)))\n        pred[~tr]=np.argmax((F[~tr]-Fm)@Wt,1)\n    return (pred==y).mean()\nprint('linear Z', [round(ridge_plain(Z,l),4) for l in [100,300,1000,3000]])\nfor sig in [0.25,0.5,1.0,2.0]:\n    rng=np.random.default_rng(0)\n    Wr=rng.normal(0,sig/np.sqrt(768),(768,768))\n    F=np.tanh(Z@Wr.T)\n    print(sig,[round(ridge_plain(F,l),4) for l in [30,100,300,1000]],flush=True)"
}
```


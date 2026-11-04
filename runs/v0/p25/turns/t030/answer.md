## reasoning summary

I'm finding that the RBF kernel on the label-free sum2 features reaches about 0.982, and the layer-normalized version comes in at 0.981. Standardizing each dimension before the RBF seems to matter, and earlier the narrower kernels with gmul at 1 and 0.5 did worse than gmul=2. Now I'm moving on to approximating the RBF with random features: a 768-dimensional pooler with tanh.

The random Fourier view doesn't carry over cleanly, since the pooler uses tanh where RBF would need cosine features. Tanh features approximate an arcsine-type kernel instead, which behaves similarly to RBF for a suitable scaling. Only 768 final features are available, so the approximation is capped there, though it may still capture most of the gain with 1200 training samples.

I'm working out how to standardize the random features: folding the per-dimension mean and std of the pooled input into the weights and bias, so the tanh layer sees standardized inputs. I'm also thinking about scale, aiming for a Gaussian weight spread near σ ≈ 1–2 so the tanh is nonlinear without saturating, and I'm weighing a random bias around N(0, 1). The next step is to test these random tanh features with ridge regression offline.

I'm going to test random tanh features on the standardized LN(sum2) with ridge regression, sweeping the weight scale, bias scale, and regularization over several seeds to compare against the RBF result. I'll then check the best setting against the real reads in two passes.



RBF on the label-free superposition x = cls11 + 2·mean11 reaches about 98.1–98.3%, versus roughly 97.5% for linear. Next I'll check offline how well 768 random tanh features followed by ridge recover that across a few weight scales.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nx=c11+2*m11; xc=x-x.mean(1,keepdims=True); n=xc/xc.std(1,keepdims=True)\nmu=n.mean(0); sd=n.std(0)+1e-6; Z=(n-mu)/sd\nT=np.eye(8)[y]\ndef ridge_plain(F,lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; Fm=F[tr].mean(0); A=F[tr]-Fm\n        Wt=np.linalg.solve(A.T@A+lam*np.eye(F.shape[1]),A.T@(T[tr]-T[tr].mean(0)))\n        pred[~tr]=np.argmax((F[~tr]-Fm)@Wt,1)\n    return (pred==y).mean()\nfor sig in [0.5,1.0,1.5,2.5]:\n    for bs in [0.0,1.0]:\n        res=[]\n        for seed in range(2):\n            rng=np.random.default_rng(seed)\n            Wr=rng.normal(0,sig/np.sqrt(768),(768,768)); br=rng.normal(0,bs,768)\n            F=np.tanh(Z@Wr.T+br)\n            res.append([round(ridge_plain(F,l),4) for l in [0.1,1,10]])\n        print(sig,bs,res,flush=True)"
}
```


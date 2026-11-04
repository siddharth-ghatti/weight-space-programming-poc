## reasoning summary

I'm settling on c=2 and β=1, which puts the robust region around 0.983. Next I want to check a different fold split and try 10-fold to make sure it isn't overfit to the fold choice, then move on to the layer 11 implementation: attention with Q=0, V=I, and W_O scaled by 2.

I'm laying out the layer 11 feed-forward block: the first projection puts each of the 1496 neurons at a training-sample center, with weights scaled by β/768 over the standardized features and a bias that cancels the mean, while the unused neurons get zero weights so their GELU output is zero. Next I'm working out how the output projection writes the eight ridge-fit scores back into the residual stream, and whether a large scale on those directions would survive the LayerNorm that follows.

I'm working out how to make the 8 class scores survive the final LayerNorm: writing them into 8 residual directions orthogonal to the all-ones vector, so the mean-centering leaves them untouched. Next I'm checking how the norm of the output depends on the scores, to see whether the normalization stays invariant to the per-sample scale.

I'm working out how the 8 class scores pass through the LayerNorm and pooler: if they sit on directions orthogonal to the all-ones vector and are scaled large enough, the argmax should come down to the scores alone.

I'm checking that the scaled class scores still dominate the small U^T n term, so the argmax follows the scores alone. Making the scale S large, around 50, should keep the logits ordered by s even though n varies per sample.

I'm checking whether the pooled output stays monotonic in the class scores: the tanh is applied per coordinate, so the argmax over classes survives it. The remaining question is how large ε can be before the tanh saturates and breaks that ordering.

I'm checking whether the pooled units can carry the argmax on their own: with the classifier as identity and zero bias, the monotone tanh keeps the ordering of the class coordinates. The open question is how to keep the residual term U^T n from interfering, and one option is to pick directions where n barely varies and cancel their constant part through the FFN output bias.

I'm now checking whether the final LayerNorm and pooler still preserve the class ordering if the residual term stays small. The idea is to subtract the constant part through the FFN output bias and rely on the monotone per-coordinate tanh to keep the argmax on the scores. I'm also wondering whether the linear term in the GELU features is enough to handle this without extra directions.

I'm checking whether the float32 precision holds up in the FFN: if the output weights are very large and the activations cancel, the errors could blow up. I want to see how big the ridge coefficients get at λ=0.01 before settling on a value that stays numerically stable.

I'm going to check whether float32 rounding of the ridge weights and GELU activations shifts the predicted logits, comparing against a float64 reference. I also want to confirm the exact GELU form the model uses, since a tanh approximation would only differ slightly. Before fixing β and λ, I'll repeat the check with a different fold seed and 10-fold.

Next I'm checking whether the result holds up with a second fold seed and a 10-fold split, and I want to compare it against an alternative that adds an explicit linear term.



Best region is c=2, β≈1, λ≈0.003–0.03, around 98.3–98.5% honest CV. Next I'll check that choice on 5- and 10-fold splits and test float32 sensitivity.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nfrom scipy.special import erf as serf\nD=np.load('base_feats.npz'); y=D['y']\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nT=np.eye(8)[y]\ngelu=lambda u: 0.5*u*(1+serf(u/np.sqrt(2)))\nx=c11+2*m11; xc=x-x.mean(1,keepdims=True); n=xc/xc.std(1,keepdims=True)\nZ=(n-n.mean(0))/(n.std(0)+1e-6); G=Z@Z.T/768\ndef mkfolds(seed,k):\n    rng=np.random.default_rng(seed); f=np.zeros(len(y),int)\n    for c in range(8):\n        idx=np.where(y==c)[0]; rng.shuffle(idx); f[idx]=np.arange(len(idx))%k\n    return f\ndef feat_cv(fn, lam, folds):\n    pred=np.zeros(len(y),int)\n    for f in range(folds.max()+1):\n        tr=folds!=f; P=fn(G[:,tr]); Ptr=P[tr]; pm=Ptr.mean(0); A=Ptr-pm\n        a=np.linalg.solve(A@A.T+lam*np.eye(len(A)),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax((P[~tr]-pm)@(A.T@a),1)\n    return (pred==y).mean()\n# linear baseline for same folds\ndef lin_cv(folds):\n    X=np.hstack([c11,m11]); return cv(X,y,folds,ridge_fit,100)[0]\nfor seed,k in [(11,5),(12,10)]:\n    fo=mkfolds(seed,k)\n    print('seed',seed,'k',k,'linear concat',round(lin_cv(fo),4))\n    for beta in [0.75,1.0,1.25]:\n        print('  beta',beta,[round(feat_cv(lambda g,b=beta: gelu(b*g),l,fo),4) for l in [0.003,0.01,0.03]],flush=True)"
}
```


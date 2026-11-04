## reasoning summary

I'm checking the scaling on the random features, since they're doing worse than plain linear ridge. With the features divided by sqrt(d), the tanh with scale 1 is nearly linear, which would explain the 95% result against the 96.2% baseline, so the ridge penalty on 768 features may be the real difference.

I'm working out whether the pooler's tanh can imitate the exponential in an RBF kernel. For a large negative argument, tanh(x) is roughly -1 + 2e^(2x), so a unit with a suitably negative bias behaves like an exponential over the range of dot products I care about, which would let me build a Nyström-style approximation from tanh units.

I'm checking whether tanh units with a large negative bias can stand in for the exponential in an RBF kernel. For a large negative argument, tanh(x) + 1 is about 2e^(2x), and float32 precision is still fine at that scale. I'm planning to test a Nyström-style fit with 768 centers and ridge on the kernel features before committing to the tanh version.

I'm checking whether the scaling explains the gap: the pooler's tanh stands in for the exponential in an RBF kernel, and the per-sample norm factor just rescales all logits, so the argmax is unchanged as long as the offset is handled. I'm weighing whether to center the targets with a tm offset as I did in kernel ridge.

Folding the exp(-γ||c_i||²) term into the coefficients keeps the tanh units in their exponential regime without changing the argmax. The per-sample norm factor and the balanced class offset are shared across classes, so the prediction stays the same as the RBF kernel ridge.

I'm checking the tanh offset: for x around -3, tanh(x)+1 is about 0.01, which matches 2e^(2x), so the bias should keep each unit in its exponential range across the data.

I'm checking the precision of the tanh features near -1: at tanh(-3.5) the representable difference is small but still enough. Since I'm fitting ridge directly on the tanh outputs, any distortion from the exponential approximation gets absorbed, so the real requirement is just that the bias keeps each unit in its exp-like range.

I'm trying a grid over the scale and bias of the tanh features, comparing exp-like features against the 768-center ridge. The earlier center features at scale 1 with zero bias came out worse than linear, which suggests the lambda grid was poorly tuned, so I'll rerun the comparison with a wider grid and check whether standardizing the features first helps.



Raw tanh random features did worse than plain linear ridge, so that setup was flawed. I'll test a Nyström-style kernel pooler with 768 centers instead, since a tanh unit with a strongly negative bias can approximate the exponential kernel.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef center_cv(X, y, nunits, gamma, lams, feat='exp', bias=-3.0, k=5, seed=0, rs=0):\n    n=len(y); S={l:np.zeros((n,8)) for l in lams}\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n        rng=np.random.default_rng(rs)\n        ci = rng.choice(tr,nunits,replace=False) if nunits<len(tr) else tr\n        Cc=Z[ci]\n        if feat=='rbf':\n            sq=(Z**2).sum(1); Hf=np.exp(-gamma*np.maximum(sq[:,None]+(Cc**2).sum(1)[None]-2*Z@Cc.T,0))\n        elif feat=='exp': Hf=np.exp(2*gamma*Z@Cc.T)\n        elif feat=='tanh': Hf=np.tanh(gamma*Z@Cc.T+bias)+1\n        Ht=Hf[tr]; T=np.eye(8)[y[tr]]-1/8\n        G=Ht.T@Ht; sc=np.trace(G)/G.shape[0]\n        for l in lams:\n            Wt=np.linalg.solve(G+l*sc*np.eye(Ht.shape[1]),Ht.T@T); S[l][te]=(Hf@Wt)[te]\n    return {l:round((S[l].argmax(1)==y).mean(),4) for l in lams}\nlams=(1e-4,1e-3,1e-2,1e-1)\nfor L in [0,1,9]:\n    X=LNr(M[L])\n    print(L,'rbf 768',center_cv(X,y,768,0.4,lams,feat='rbf'))\n    print(L,'rbf all',center_cv(X,y,5000,0.4,lams,feat='rbf'))\n    print(L,'exp 768',center_cv(X,y,768,0.4,lams,feat='exp'))\n    print(L,'tanh 768',center_cv(X,y,768,0.4,lams,feat='tanh'))\n"
}
```


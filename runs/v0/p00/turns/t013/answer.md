## reasoning summary

I'm checking the fit: weighted relative error is about 0.2% over the K tokens, and the unweighted mean is dominated by the 339 neuron-less tokens, which is fine. Then I'm settling on β=10 and θ=0.6, since they give few random outputs and near-zero special ones. I want to confirm the model's GELU is the exact erf form rather than the tanh approximation, because my fit assumes erf.

I'm checking the layer-norm epsilon, token type ids, and position ids against the BERT config, and they all look consistent with what the model computes. Next I want to look at the magnitudes of the pre-activations over the full vocabulary, to make sure the unused tokens with large whitened norms don't produce big spurious outputs. Then I'll pick a scale factor for the residual based on how the norms of the sampled tokens are distributed.

I'm sizing the residual scale so the aggregated norms stay around 3, and I'll check the distribution of the sampled token norms to choose it. I'm also weighing whether ridge scores or NB scores work better as the target, since the CV gap is only a few sentences and could be noise.

I'm comparing a few scoring variants against each other: NB with a different smoothing alpha, ridge with a couple of regularization strengths, and an average of NB and ridge. Binary presence features would need deduplicated tokens, which the network can't produce, so I'm keeping the score as a sum over token counts with repetition.



β=10 and θ=0.6 give about 0.2% weighted error on practice tokens, with near-zero output for special tokens. Next I'll compare closed-form scoring options with data-only CV and check score magnitudes before fixing the table.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522\nn = np.array([len(t) for t in ids]); Xtf = sp.diags(1.0/n)@X\nres = {}\nfor seed in [0,1,2]:\n    rng=np.random.RandomState(seed); perm=rng.permutation(len(P)); folds=np.array_split(perm,5)\n    def nbS(tr,alpha):\n        cnt=np.zeros((V,8))\n        for c in range(8): cnt[:,c]=np.asarray(X[tr][y[tr]==c].sum(0)).ravel()\n        lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*V); return lp-lp.mean(1,keepdims=True)\n    def ridgeS(tr,lam):\n        Xc=Xtf[tr]; Yc=np.eye(8)[y[tr]]\n        Kk=(Xc@Xc.T).toarray(); A=np.linalg.solve(Kk+lam*np.eye(len(tr)), Yc-Yc.mean(0))\n        S=np.asarray(Xc.T@A); return S-S.mean(1,keepdims=True)\n    for name,fn in [('nb0.1',lambda tr:nbS(tr,0.1)),('nb0.3',lambda tr:nbS(tr,0.3)),('nb0.5',lambda tr:nbS(tr,0.5)),\n                    ('ridge1e-3',lambda tr:ridgeS(tr,1e-3)),('ridge3e-3',lambda tr:ridgeS(tr,3e-3)),('ridge1e-2',lambda tr:ridgeS(tr,1e-2)),\n                    ('blend',lambda tr: nbS(tr,0.3)/4 + ridgeS(tr,3e-3)*1.0)]:\n        pred=np.zeros(len(P),int)\n        for f in folds:\n            tr=np.setdiff1d(np.arange(len(P)),f); S=fn(tr)\n            pred[f]=(X[f]@S).argmax(1)   # sum = n*mean; argmax same w/o bias\n        res.setdefault(name,[]).append((pred==y).mean())\nfor k,v in res.items(): print(k, np.round(v,4), np.mean(v))\n# magnitude of NB scores\nS=np.load('S_nb.npy'); K=np.load('K.npy')\nnr=np.linalg.norm(S[K],axis=1); print('||S|| pct', np.percentile(nr,[50,90,99,100]), 'max entry', np.abs(S[K]).max())"
}
```


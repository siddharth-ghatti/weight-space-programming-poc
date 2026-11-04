## reasoning summary

Mean-pooled features hold up at nearly every layer, with even the bag-of-embeddings layer 0 reaching about 96%, so the task looks largely keyword-driven. Next I want to test concatenations of layers, such as mean layer 0 with mean layer 12, or mean layer 6 with layer 12, to see whether combining them helps.

Now I'm working out how to get the mean of the final layer's output into the CLS token. Aggregating in layer 11 only gives the mean of its input, and nothing after the last layer can attend over the tokens, so that exact feature looks out of reach unless the computation is restructured somehow.

I'm weighing whether to compare mean features from earlier layers, ≤11, together with CLS at several layers, since the noise in these results is about half a percent and smaller gaps won't mean much. I'm also considering whether to aggregate into CLS at an intermediate layer, though the later layers would then process CLS as usual and distort it.

I'm weighing whether attention pooling or downweighting special tokens like [CLS], [SEP], and punctuation could beat simple averaging, though those gains would probably be small. I'll first check the headroom by testing concatenations such as mean layer 0 with mean layer 11, and if that gain turns out under about 1%, I'll stick with the simpler approach.

I'm going to check whether ridge still holds up without standardization, since the real model features are normalized per sample. Then I'll run the combination tests like mean layer 0 with mean layer 11 to see if the gain is worth the extra complexity.



Mean-pooled features reach about 95–96.5% CV accuracy at every layer, even layer 0, so the task leans heavily on vocabulary, and the best mean I can route into [CLS] is L11 at about 96%. I'm checking whether combining layers or other variants gives real gains beyond the roughly 0.5% noise before building.



## tool_use: run_python

```json
{
 "code": "lab = np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nM=np.load(WORK/'base_mean.npy'); C=np.load(WORK/'base_cls.npy')\ndef ridge_cv(X, lams, std=True):\n    res={}\n    for lam in lams:\n        pred=np.zeros(N,int)\n        for f in range(5):\n            tr=folds!=f; te=~tr\n            mu=X[tr].mean(0); sd=(X[tr].std(0)+1e-6) if std else np.ones(X.shape[1])\n            Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n            # dual form for speed when d>n\n            if Xt.shape[1]>Xt.shape[0]:\n                K=Xt@Xt.T; a=np.linalg.solve(K+lam*np.eye(len(K)), Y[tr]-Y[tr].mean(0)); S=Xe@(Xt.T@a)\n            else:\n                Wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(Xt.shape[1]), Xt.T@(Y[tr]-Y[tr].mean(0))); S=Xe@Wt\n            pred[te]=np.argmax(S,1)\n        res[lam]=round((pred==lab).mean(),4)\n    return res\ndef ln(X): \n    X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ncombos={'m11':[M[11]],'m11_ln':[ln(M[11])],'m0+m11':[M[0],M[11]],'m6+m11':[M[6],M[11]],'m0+m6+m11':[M[0],M[6],M[11]],\n        'm11+c11':[M[11],C[11]],'m11+c12':[M[11],C[12]], 'm12':[M[12]], 'm0+m12':[M[0],M[12]], 'all_mean':[M[l] for l in range(0,13,2)]}\nfor k,v in combos.items():\n    X=np.concatenate(v,1)\n    print(k, ridge_cv(X,[30,100,300,1000]), flush=True)\nprint('m11 nostd', ridge_cv(M[11],[1,10,100],std=False))\n"
}
```


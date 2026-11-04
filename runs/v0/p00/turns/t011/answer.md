## reasoning summary

The crosstalk between days and weeks is semantic, which should help with generalization. Next I'm considering the FFN output design, a GELU neuron with a threshold around 0.55 to 0.65, where weeks would partly fire at about 0.57 similarity to days. That's fine provided weeks shares days' class profile, but I still need to check whether it does.

I'm weighing a linear alternative, where the response itself acts as a smooth similarity kernel, and fitting output weights over the known tokens by ridge least squares. I'm leaning toward excluding random unseen vocabulary from the fit, though forcing unseen tokens to zero might still matter for semantically close words like "weeks."

I'm working out the output scale: with a threshold of 0.5 and a sharpness of 8, a token's own neuron gives about 3.8, cross-talk at 0.5 is zero, and very similar tokens around 0.6 to 0.75 fire only partly. That keeps the fit matrix close to diagonal, so it should be well conditioned. The open question is whether to add a second FFN layer, since there are about 3,411 known tokens but only 3,072 neurons.

I'm checking whether the layer-1 detectors need to be zero on the DF dimensions. The DF coordinates of h1 hold the scores, and since C is blocked with a regularized identity on those dimensions, the fitted detector weights there should come out zero, which is what I want.

I'm considering whether the extra 339 rare tokens are worth including in the output layer. Their scores are only weakly informative, and the cross-validation gap is negligible, so I'm leaning toward keeping layer 0 to the 3,072 most frequent tokens.

I'm wondering whether bigram features could lift the 96% cross-validation accuracy by a point or two. That would mean a previous-token head in layer 0, but position information is mixed into the same residual stream as token identity, so that would be complicated to build.

I'm checking whether bigram features would lift the 96% accuracy, which would need a previous-token head in layer 0. Position information is mixed into the same residual stream as the token identity, so building that attention pattern from learned position embeddings would be hard. I'll measure the potential gain before deciding whether it's worth pursuing.



The crosstalk looks semantic and should help generalization. I'm checking with data-only CV whether bigrams add enough to justify a previous-token attention head.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\n# bigram features via hashing pairs into dict\nbg = {}\nrows=[];cols=[]\nfor i,t in enumerate(ids):\n    for a,b2 in zip(t[:-1],t[1:]):\n        key=(a,b2)\n        if key not in bg: bg[key]=len(bg)\n        rows.append(i);cols.append(bg[key])\nXB = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(P),len(bg)))\nXU = X[:, np.unique(X.indices)]\nrng = np.random.RandomState(0); perm = rng.permutation(len(P)); folds = np.array_split(perm,5)\ndef nbcv(Xf, alpha=0.3, w=None):\n    pred=np.zeros(len(P),int)\n    Vf = Xf.shape[1]\n    for f in folds:\n        tr=np.setdiff1d(np.arange(len(P)),f)\n        cnt=np.zeros((Vf,8))\n        for c in range(8): cnt[:,c]=np.asarray(Xf[tr][y[tr]==c].sum(0)).ravel()\n        lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*Vf); lp-=lp.mean(1,keepdims=True)\n        pred[f]=(Xf[f]@lp).argmax(1)\n    return (pred==y).mean(), pred\nfor alpha in [0.1,0.3,1.0]:\n    print('uni',alpha, nbcv(XU,alpha)[0], 'uni+bi', nbcv(sp.hstack([XU,XB]).tocsr(),alpha)[0])\nacc,pred = nbcv(XU,0.3)\ncm = np.zeros((8,8),int)\nfor a,b2 in zip(y,pred): cm[a,b2]+=1\nprint(LABELS); print(cm)"
}
```


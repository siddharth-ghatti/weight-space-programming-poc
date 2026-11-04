## reasoning summary

I'm picking eight dimensions with low variance across the vocabulary, since they should carry little token information. A couple of the candidates have fairly large means, so zeroing them shifts the vector's spread a bit, but that seems acceptable. I'll settle on 145, 175, 529, 121, 216, 654, 105, and 353, and honestly the exact choice doesn't matter much.

Next I'm setting up the detector: I'll standardize each hidden state over all 768 dimensions first, then zero out the chosen dimensions with gamma and beta set to zero. After that I want to see how much position changes the representation, comparing a token at position 1 against the same token at position 50. I also plan to build the token set from the frequent vocabulary, leaving out the special tokens, and check whether naive Bayes accuracy holds up when restricted to the top-N tokens by count.



I'm reserving the 8 lowest-variance dimensions as feature dims, then checking how CV accuracy changes when only the most frequent tokens get scores and how position affects layer-0 representations.



## tool_use: run_python

```json
{
 "code": "import scipy.sparse as sp\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])\nids = [t['ids'] for t in tokenize([p['text'] for p in P])]\nV=30522\nrows=[];cols=[]\nfor i,t in enumerate(ids):\n    for j in t: rows.append(i);cols.append(j)\nX = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(P),V))\nrng = np.random.RandomState(0); perm = rng.permutation(len(P)); folds = np.array_split(perm,5)\ndef nbfit(Xc,yc,alpha=0.3,keep=None):\n    cnt = np.zeros((V,8))\n    for c in range(8): cnt[:,c] = np.asarray(Xc[yc==c].sum(0)).ravel()\n    lp = np.log(cnt+alpha) - np.log(cnt.sum(0)+alpha*V)\n    lp -= lp.mean(1,keepdims=True)\n    if keep is not None:\n        mask = np.zeros(V,bool); mask[keep]=True; lp[~mask]=0\n    return lp\ntot = np.asarray(X.sum(0)).ravel()\nfor N in [1000,1500,2000,2500,3000,3413]:\n    accs=[]\n    pred=np.zeros(len(P),int)\n    for f in folds:\n        tr=np.setdiff1d(np.arange(len(P)),f)\n        trc = np.asarray(X[tr].sum(0)).ravel()\n        keep = np.argsort(-trc)[:N]\n        S = nbfit(X[tr],y[tr],keep=keep)\n        pred[f]=(X[f]@S).argmax(1)\n    print(N,(pred==y).mean())\n# position sensitivity\nE = base('bert.embeddings.word_embeddings.weight'); Pp = base('bert.embeddings.position_embeddings.weight'); T = base('bert.embeddings.token_type_embeddings.weight')\ng = base('bert.embeddings.LayerNorm.weight'); b = base('bert.embeddings.LayerNorm.bias')\ndef ln(x, gg, bb, eps=1e-12):\n    m = x.mean(-1,keepdims=True); v = ((x-m)**2).mean(-1,keepdims=True)\n    return (x-m)/np.sqrt(v+eps)*gg+bb\nDf = np.array([145,175,529,121,216,654,159,541])\ngam = np.ones(768,np.float32); gam[Df]=0\ndef h0(tokids, pos):\n    x0 = ln(E[tokids] + Pp[pos] + T[0], g, b)\n    return ln(x0, gam, np.zeros(768,np.float32))\nvv = vocab()\nfor w in ['deductible','premium','means','cancel','excluded']:\n    k = vv[w]\n    hs = np.stack([h0(np.array([k]), p)[0] for p in [1,5,10,20,40,60]])\n    hn = hs/np.linalg.norm(hs,axis=1,keepdims=True)\n    print(w, np.round(hn@hn.T,3)[0])"
}
```


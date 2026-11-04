## reasoning summary

I'm turning to robustness and improvements now. The unigram classifier's 97.3% CV figure holds up for practice-like data, which is good. For the bigram side, I want to see how well position can be decoded from h0 and h1, and I'm considering a cleaner previous-token head that doesn't depend on linearly decoding position from a noisy h0.

I'm working through how position gets encoded in h0: the position term gets scaled by a token-dependent factor, so a linear probe for position is noisy. I'm now looking at designing the query projection to pull out the position component while suppressing the token component.

I'm checking whether position stands out from token variation in the hidden states, using a generalized eigenproblem between position and token covariances. If the top directions have a large ratio, position could be decoded cleanly, and I'd fit a bilinear score to peak at the previous position.

I'm now setting up a bilinear score so that the previous-position key gets the highest match, fitting the coefficients by least squares over positions 0 to 80. I'm also expecting the token-dependent scaling and leftover token components to add noise, and the margin needed for reliable attention is demanding. Verifying it would mean computing scores for specific token pairs at adjacent positions.

I'm weighing whether checking attention at the token-pair level counts as analysis of a designed component on synthetic inputs rather than running the model on data. Then I'm turning to the bigram detectors: the previous-token head would need to carry enough identity information for the layer-1 FFN to detect pairs like "not" followed by "covered," so copying a lower-dimensional projection of the previous token's hidden state looks like a cheaper option than copying its class scores.

I'm considering whether a more useful gain would come from better unigram weights rather than bigram detectors, since the ridge-versus-naive-Bayes gap suggests the weight table alone moves accuracy by about a point. I'm also wondering whether the bigram cases that matter most, like negation phrases, can be captured more cheaply than through a full previous-token head.

I'm considering whether a non-uniform attention pooling, where CLS weights tokens by learned importance so stopwords get down-weighted, could beat plain mean pooling, though fitting that would be iterative and I'm leaning toward skipping it. I'm also weighing whether sublinear term frequency is even possible here, and I'm wondering what the sentence length feature would contribute.

I'm checking whether a bias term would help the classifier, since the argmax without one already reaches about 97%. One way to get a bias is through the CLS self term, though the nonlinearity makes that only approximate, so I want to test it directly.

I'm going to check whether adding an intercept to the ridge model improves cross-validation, and I'll try different feature normalizations too. After that I'll look at the misclassified examples under the blend to see what errors remain.



v1 is saved; next I'll check what the unigram table still gets wrong in CV and whether a closed-form ridge intercept helps, before deciding on bigrams.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522; N=len(P)\nn=np.array([len(t) for t in ids]); Xtf=sp.diags(1.0/n)@X\ndef tables(tr, alpha=0.3, lam=3e-3):\n    cnt=np.zeros((V,8))\n    for c in range(8): cnt[:,c]=np.asarray(X[tr][y[tr]==c].sum(0)).ravel()\n    lp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*V); Snb=lp-lp.mean(1,keepdims=True)\n    Xc=Xtf[tr]; Yc=np.eye(8)[y[tr]]\n    Kk=(Xc@Xc.T).toarray(); A=np.linalg.solve(Kk+lam*np.eye(len(tr)), Yc-Yc.mean(0))\n    Sr=np.asarray(Xc.T@A); \n    b = Yc.mean(0) - np.asarray(Xc.mean(0)).ravel()@Sr\n    Sr-=Sr.mean(1,keepdims=True)\n    return Snb, Sr, b\npreds={'blend':np.zeros(N,int),'blend+b':np.zeros(N,int)}\nrng=np.random.RandomState(0); perm=rng.permutation(N); folds=np.array_split(perm,5)\nfor f in folds:\n    tr=np.setdiff1d(np.arange(N),f)\n    Snb,Sr,b=tables(tr)\n    S=Snb/4+Sr\n    preds['blend'][f]=(Xtf[f]@S).argmax(1)\n    preds['blend+b'][f]=(Xtf[f]@S+b).argmax(1)\nfor k,v in preds.items(): print(k,(v==y).mean())\npred=preds['blend']\nwrong=np.where(pred!=y)[0]\nfor i in wrong[:40]:\n    print(f'[{LABELS[y[i]][:10]} -> {LABELS[pred[i]][:10]}] {P[i][\"text\"]}')"
}
```


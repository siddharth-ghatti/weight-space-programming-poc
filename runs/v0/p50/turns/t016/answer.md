## reasoning summary

I've confirmed the construction matches, with cosine similarity around 0.9999997, and that mean pooling includes the CLS and SEP tokens just as the attention does. That means my numpy predictions of the final features from the base reads should be reliable. Next I'm looking at improvements to the classifier, weighing closed-form options like LDA with shrinkage, ridge without standardization, or nearest class mean, since iterative logistic regression is ruled out. I'm also starting to think about multi-layer pooling.

I'm weighing how to get multi-layer pooling into the CLS token. The idea is to have an earlier layer's CLS attention pull in the mean, but changing CLS at that layer would also shift how other tokens attend to it, so I'd need to re-measure the downstream features.

I'm weighing a parallel path that would carry earlier layers' means forward through the token residual stream, though mean11 already holds some of that information, transformed nonlinearly. I'm also considering whether non-uniform attention in layer 11 could pool differently, but since every head only sees h11, that doesn't look promising. The more concrete option now is a closed-form readout built on the layer-11 FFN, a random-feature style setup that approximates the small MLP that follows the pooled features.

I'm weighing a way to get the FFN's hidden activations without simulating the whole model in numpy, since reimplementing the forward pass is off-limits. Computing a GELU over a linear map of my own features looks like it could be the same thing in disguise. Another route would be to read the pooled output, though that only gives the 768-dim pooler result rather than the 3072 intermediate units.

I'm weighing whether to put the pooler to work as a random-feature layer, since its output is directly readable. Setting a designed weight matrix and reading the pooled vectors would give a 768-unit tanh layer that a ridge fit could use in closed form. I'd still have to judge whether that beats the linear readout, and I'd rather check it with reads than by simulating the pooler in numpy.

I'm checking which of the roughly 3% errors are confused with which classes, to see whether they look like ambiguous cases, and I'm planning to try a few closed-form classifiers on the features.

I'm going to compare LDA with shrinkage, ridge, and a whitened-PCA classifier against the current 96.9% ridge baseline. I'm also thinking about how to get early-layer pooling into the CLS token, though changing CLS at an earlier layer would shift what the later tokens attend to.

I'm considering whether the CLS query could be shrunk toward zero at an early layer so its attention averages over all tokens, though this only works cleanly if the CLS hidden state barely varies across sentences, which holds at the earliest layers but not clearly at layer 1.

I'm considering whether to pool across heads at layer 11 with different attention patterns, though every head only sees h11. One option is attention pooling with a fixed query vector, so each head takes a softmax-weighted mean with weights driven by a learned direction in the token space.

I'm considering giving each head its own attention pattern, with the query fixed so each head favors tokens that point toward a class-specific direction. That would work like keyword-focused pooling rather than a plain mean, and each head's 64-dim output could be projected from the weighted mean to build the 768-dim feature vector.

Thinking about how to combine the layers, I'm weighing uniform mean pooling across six heads against keyword-focused attention pooling in the other six. Evaluating either one needs per-token hidden states, and simulating the attention in numpy would amount to reimplementing it, so I want to see the error patterns first before choosing.



The [CLS] construction checks out (cosine 0.9999997 against the normalized mean of layer 11), and ridge CV on these features gives about 96.9%. Before building the head, I'm examining the errors and trying shrinkage LDA, unstandardized ridge, and nearest-centroid.



## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy'); E=np.load('feats/edA_cls12.npy')\nP=practice()\ndef fit_ridge(X,y,lam):\n    mu=X.mean(0); sd=X.std(0)+1e-6; Xs=(X-mu)/sd\n    Y=np.eye(8)[y]-1/8\n    Wt=np.linalg.solve(Xs.T@Xs+lam*np.eye(X.shape[1]), Xs.T@Y)\n    return Wt/sd[:,None], -(mu/sd)@Wt\ndef fit_lda(X,y,shr):\n    mus=np.stack([X[y==k].mean(0) for k in range(8)]); Xc=X-mus[y]\n    S=Xc.T@Xc/len(y); S=(1-shr)*S+shr*np.trace(S)/S.shape[0]*np.eye(S.shape[0])\n    Wt=np.linalg.solve(S,mus.T); b=-0.5*np.sum(mus.T*Wt,0)\n    return Wt,b\ndef cvf(X,y,fitter,arg):\n    pr=np.zeros(len(y),int); sc=np.zeros((len(y),8))\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Wt,b=fitter(X[tr],y[tr],arg); sc[te]=X[te]@Wt+b; pr[te]=sc[te].argmax(1)\n    return (pr==y).mean(), pr\nfor s in [0.01,0.03,0.1,0.2,0.4]: print('lda',s,cvf(E,lab,fit_lda,s)[0])\nacc,pr=cvf(E,lab,fit_ridge,120); print('ridge',acc)\nC=np.zeros((8,8),int)\nfor a,b in zip(lab,pr): C[a,b]+=1\nprint(C)\nfor i in np.where(pr!=lab)[0][:45]:\n    print(LABELS[lab[i]][:10],'->',LABELS[pr[i]][:10],'|',P[i]['register'][:8],'|',P[i]['text'][:150])\n"
}
```


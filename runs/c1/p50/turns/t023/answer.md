## reasoning summary

I'm seeing a noticeable learning curve: accuracy climbs from 97.6% at 5-fold to 98.2% at 20-fold, and the full-data estimate lines up with the 98.31% test result, so cross-validation looks like a good proxy. That makes me think a target of roughly 98.0–98.2% CV accuracy would give some margin above the 97.8% threshold.

Comparing the 20-fold CV estimate of 98.2% against the 98.31% test result, I'd aim for CV of at least 98.0%, with some risk, since test-set noise is still unknown. Shallow RBF features already reach about 97.8% at 10-fold, so better nonlinear or attention-based features might push that past 98%. Next I'm weighing the cheapest model shapes and which features each one affords, starting with the one-layer and two-layer options.

I'm weighing a few ways to build the features: a plain mean over the full 768-dim embedding output, or a compressed version projected onto the top discriminative directions so some attention heads stay free for weighted pooling. On top of that I'd add a nonlinear FFN on the CLS token and a tanh pooler, then fit the classifier in closed form, possibly with random features to approximate an RBF kernel.

I'm weighing whether a Nyström-style pooler could work better than random Fourier features here. Anchoring the tanh units on training examples would give kernel-like features, though 768 anchors fall short of the 1496 needed for an exact kernel machine. The FFN units might add more anchors if I use GELU on them.

I'm checking whether a tanh pooler can approximate the RBF kernel: after layer norm, the RBF similarity is an exponential of a dot product, which tanh can't produce directly, though GELU units in the FFN might come closer. I'm also considering a two-layer alternative where a tanh hidden layer on whitened random directions stands in for the kernel features.

I'm checking whether random features from the mean-pooled hidden states, passed through tanh and fit with ridge, can match kernel ridge regression. I'd test that offline before building the full pooler. I'm also wondering whether attention pooling across several heads could capture the hard cases, like negations and conditional phrasing, that a plain mean tends to blur.

I'm weighing a two-layer design: one layer to contextualize tokens, then a pooling layer with a nonlinear head, using mean hidden states and the CLS vector as features. The shallow RBF baseline sits at 97.77%, against 97.19% for the single layer, so the single-layer route would need an extra 0.6–1% from attention tricks to catch up.

I'm now considering whether a two-layer model with the pooling layer at the top could be built with an FFN of around 1024 or 3072 units, which would change the parameter cost noticeably. My first experiment would keep the full 3072-unit FFN in layer 0 and reuse the starting model's pooling layer as the top layer.

I'm going to test the nonlinear head first, offline: tanh random features on the layer-0 and (0,1) mean features, fit with ridge, and compare against the RBF baseline. After that I'll design a pooling layer with several attention patterns, using per-class token scores to shape them.

I'm thinking about whether the 8 class-score heads and the 4 mean-feature heads together can cover the 768 dimensions, since each head's 64 dimensions have to fit inside the 768-dim output. I also want to see whether a PCA-256 projection of the mean features keeps enough accuracy, so I'll check that with ridge CV.

I'm going to check how well ridge and RBF hold up when the mean features are compressed with PCA to 64, 128, 256, and 384 dimensions. If accuracy stays close to the full 768-dim result, a few heads could be freed up for other pooling patterns.



10-20-fold CV tracks the original's 98.31% test score, so I'll target about 98.2% at 20-fold as the working benchmark. Shallow layer 0+1 features already reach 97.8% at 10-fold, so next I'll test offline whether the pooler's tanh random features can match the RBF kernel and how much PCA-compressed mean features lose.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef tanh_rf_cv(X, y, nfeat=768, scale=1.0, lams=(0.1,1,10), k=10, seeds=(0,), rs=0):\n    Xs=X-X.mean(0); Xs=Xs/np.sqrt((Xs**2).sum(1).mean())  # unit rms norm\n    rng=np.random.RandomState(rs); Wr=rng.randn(X.shape[1],nfeat)*scale; br=rng.randn(nfeat)*0.0\n    F=np.tanh(Xs@Wr+br)\n    return cv_acc(F,y,lams=lams,k=k,seeds=seeds)\nX0=M(0)\nfor sc in [1,2,4,8]:\n    print('L0 tanh rf scale',sc, tanh_rf_cv(X0,y,scale=sc))\nprint('L0 lin 10f',cv_acc(X0,y,lams=(3,),k=10))\n# PCA compressions\nXc=X0-X0.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False)\nfor r in [32,64,128,256,384]:\n    Z=Xc@Vt[:r].T\n    print('PCA',r,'lin',cv_acc(Z,y,lams=(1,3,10),k=10), 'rbf', krr_cv(rbf(Z),y,(0.01,),k=10,seeds=(0,)))"
}
```


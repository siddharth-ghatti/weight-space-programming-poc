## reasoning summary

I'm comparing the 1024-subset results across two seeds, where base M0+M9 comes out at 97.95 and the edited version at 97.66, and the edit actually edges ahead for M0+M8. The gaps are around 0.3%, which is within the noise. On the full 1496 set the scores run about 0.5% higher, so I'm checking how base M0+M9 at 98.46 relates to its 1024 figure before looking at the edited version there.

I'm checking whether the full-1496 numbers can be inferred from the 1024 subset, since the read limit means each full evaluation takes two passes. With a 31-pass budget, I'm weighing whether to improve the sequential pruning with drift correction, which would cost about ten passes, or to reuse the existing exposures in a different way.

I'm weighing whether the carried scores in the reserved dimensions at layers 2 and 6 can be kept in the features, since they were fit on all practice labels and would make a cross-validated estimate optimistic. One option is to refit the carry heads with my own ridge so the evaluation stays honest, while the other is to spend passes improving the pruning instead.

I'm sketching a kernel-style final model on the pruned features: normalizing the hidden vectors, using random training points as anchors, and applying tanh units feeding a ridge readout. I'm wondering whether the FFN in the pooling layer could carry these kernel features and write class scores into the reserved dimensions, so the pooler only has to decode them.

I'm settling on a simpler option for the pool layer: the FFN would hold about 1024 GELU units anchored on training points, writing class scores into the eight reserved dimensions, so the pooler only needs to read them. I'm checking that the layer norm doesn't wash out those scores, since the reserved dimensions would dominate the vector's standard deviation, though the argmax should still be preserved.

I'm working out how to get the pool layer's attention to produce a centered, normalized version of the mean-pooled hidden state, by setting the value and output weights so the output is c·P(mean − μ). I'm checking whether a large scale c makes the post-LayerNorm vector track the normalized centered mean closely enough.

I'm working out how the pool layer's FFN would turn each anchor's normalized features into a kernel unit, with the scale and bias chosen so the cosine similarities land in a usable range. I'm weighing whether to compute those anchor features from the actual layer-normalized outputs, which means reading the CLS hidden states, or to approximate them from the mean-pooled features alone.

I'm planning to check the pipeline's accuracy by simulating a GELU network on the pruned features, using cosine similarity to anchor points.

I'm now wondering whether M0 can be carried into the pool layer, since only eight reserved dimensions are free and the earlier layers already use them for the layer 2 and layer 6 carries. The layer 1 normalization clears those dimensions, so I'm checking what the reserved slots at layer 9 actually hold before deciding where an M0 carry could go.

I'm weighing whether M0 can be combined with M9 in a kernel, which would need a nonlinear pass over M0 at the CLS position and so looks complicated. A simpler route is to append the 8-dimensional M0 scores linearly and test a kernel ridge on M9 with those scores added.

I'm weighing whether M0 can be carried into the pooling layer alongside M9, since the pooling layer only sees the final hidden states. Base M0+M9 beats M9 alone by about 0.3% on the full set and by about 1% on the pruned 1024 subset, so M0 seems to matter more once the features are pruned. I'm thinking about whether the 12 pooling heads could expose that information, or whether a wider reserved channel would be needed to carry it.

I'm considering whether M0 could reach the pooling stage through an earlier layer, but the CLS token is attended to by every other token, so changing it at layer 0 would alter everything downstream. I could instead widen the reserved channel to around 32 dimensions to carry a PCA of M0, though that would likely hurt the content features, so I'll test it empirically with kernel ridge on M9 combined with 8, 16, and 32 PCA components of M0.



On the noisy subset, pruned M9 with M0 lands at 97.66%, near the base 97.95%, and M0 adds about 1% to the pruned features. Next I'll measure how much M0 information the final layer needs, using PCA-k or 8 out-of-fold class scores.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz'); P=practice(); id2i={p['id']:i for i,p in enumerate(P)}\nids19=json.load(open('pass_19/meta.json'))['ids']; idx=np.array([id2i[i] for i in ids19]); yy=y[idx]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\nE9=cn(np.load('pass_19/mean_hidden_L9.npy')); Z0=cn(d['mean_hidden_L0'][idx])\ndef kcv(Zfun, lam=0.005, k=10, seeds=(0,1), g=1.0):\n    Y=np.eye(8)[yy]; accs=[]\n    for s in seeds:\n        f=folds(yy,k,s); S=np.zeros((len(yy),8))\n        for i in range(k):\n            tr=f!=i; te=f==i; Z=Zfun(tr)\n            Kt=np.exp(g*(Z@Z[tr].T-1))\n            A=np.linalg.solve(Kt[tr]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=Kt[te]@A\n        accs.append((S.argmax(1)==yy).mean())\n    return round(float(np.mean(accs)),4)\nprint('M9e', kcv(lambda tr: E9))\nprint('M0+M9e', kcv(lambda tr: np.hstack([Z0,E9])/np.sqrt(2)))\nU,s,Vt=np.linalg.svd(Z0-Z0.mean(0),full_matrices=False)\nfor kk in [8,16,32,64]:\n    for beta in [0.5,1.0]:\n        def zf(tr,kk=kk,beta=beta):\n            u,s_,vt=np.linalg.svd(Z0[tr]-Z0[tr].mean(0),full_matrices=False)\n            Pk=(Z0-Z0[tr].mean(0))@vt[:kk].T; Pk/=np.sqrt((Pk[tr]**2).sum(1).mean())\n            return np.hstack([E9, beta*Pk])/np.sqrt(1+beta**2)\n        print('PCA',kk,beta,kcv(zf))\nfor beta in [0.5,1.0,2.0]:\n    def zf(tr,beta=beta):\n        W,b=ridge_fit(Z0[tr]*30,yy[tr],10.0); Sc=Z0*30@W+b; Sc-=Sc[tr].mean(0); Sc/=np.sqrt((Sc[tr]**2).sum(1).mean())\n        return np.hstack([E9,beta*Sc])/np.sqrt(1+beta**2)\n    print('linscores',beta,kcv(zf))"
}
```


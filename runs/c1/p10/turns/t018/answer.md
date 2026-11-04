## reasoning summary

Pooling the embedding-level mean together with a deeper layer's mean gives 98.3%, so the combination is helping. The open question is how the M0 part gets through the network: a carry head in layer 0 could write the mean of the embeddings into a reserved channel, but a kernel reading M0 would then need the full 768-dimensional vector at the pooler, not just 8 scalar channels.

I'm weighing whether to add a low-dimensional projection of M0 alongside M8, since carrying all 768 dimensions of M0 to the pooler isn't feasible. I'm considering two ways to do it: a kernel over M8 plus 8 out-of-fold ridge scores of M0, or an additive model that sums a kernel on M8 with a linear kernel on M0.

The additive version looks workable: the linear M0 part could ride on the existing carry heads, since layer 2 and layer 6 already write 8 ridge scores each into reserved dims, so only their projections would need refitting. The open question is whether the kernel on M8 and the linear term on those scores can be fit jointly as one ridge problem.

I'm checking whether a sum of a kernel on the last-layer mean and linear terms on the carried M2 and M6 scores can be fit as one ridge problem, with each linear block standardized before it's combined.

I'm checking whether the decoded carry values still line up once the per-sample scaling after layer norm is applied. The reserved dims hold ridge scores multiplied by each sample's own factor, so the final head should be fit on the actual read vectors rather than the projections computed beforehand. That lets it re-weight the eight carried scores freely.

I'm working through how the pooler handles the mixed inputs: a few of its units would pass the eight reserved carry dimensions through roughly linearly, while the rest read only the non-reserved dimensions of the final CLS vector. The classifier would then be fit jointly over both groups, so the kernel features and the carried linear terms get weighted together. The remaining question is how the last-layer mean makes it into that CLS vector.

I'm weighing whether the pooling layer's uniform attention with an identity value and output map can keep the reserved dimensions intact, with the layer norms only rescaling them per sample. I'll evaluate the additive models first, since a kernel on M8 plus linear terms on M2 and M6 could land near 98.3%, though that still implies a large parameter count that FFN pruning would need to bring down.

I'm now checking the shallow version of the additive model: a kernel on the layer-1 mean plus a linear term on M0, which would mean repurposing layer 0's carry head.



The 98.3% and 98.5% mixes beat single layers, but the 768-dim [CLS] can't hold two full vectors, so I'll try an additive model: a kernel on the last layer's mean plus linear ridge scores from earlier layers, carried in the reserved dimensions, and evaluate it with a sum kernel.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef zs(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef add_cv(kb, lin, y, gammas, cs, lams, seeds=(0,1,2), k=5):\n    n=len(y); res={}\n    for s in seeds:\n        for te in folds(n,k,s,y):\n            tr=np.setdiff1d(np.arange(n),te)\n            Z=zs(kb,tr); sq=(Z**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0)\n            ZL=np.concatenate([zs(X,tr) for X in lin],1)/np.sqrt(len(lin)) if lin else None\n            KL=ZL@ZL.T if lin else 0\n            T=np.eye(8)[y[tr]]-1/8\n            for g in gammas:\n                KR=np.exp(-g*D2)\n                for c in cs:\n                    K=KR+c*KL; w,V=np.linalg.eigh(K[np.ix_(tr,tr)]); R=V.T@T; Kte=K[np.ix_(te,tr)]\n                    for l in lams:\n                        key=(g,c,l); pred=(Kte@(V@(R/(w+l)[:,None]))).argmax(1)\n                        res.setdefault(key,[0,0]); res[key][0]+=(pred==y[te]).sum(); res[key][1]+=len(te)\n    return {k_:round(v[0]/v[1],4) for k_,v in res.items()}\nfor name,kb,lin in [('K8+L2L6',M[8],[M[2],M[6]]),('K8+L0',M[8],[M[0]]),('K9+L2L6',M[9],[M[2],M[6]]),('K9+L0',M[9],[M[0]]),\n                    ('K1+L0',M[1],[M[0]]),('K6+L2',M[6],[M[2]]),('K7+L2L6',M[7],[M[2],M[6]])]:\n    r=add_cv(kb,lin,y,(0.2,0.4),(0.3,1,3),(0.003,0.03))\n    best=sorted(r.items(),key=lambda t:-t[1])[:3]\n    print(name,best)\n"
}
```


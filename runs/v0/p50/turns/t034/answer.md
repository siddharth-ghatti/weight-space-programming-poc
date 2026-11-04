## reasoning summary

I'm checking whether the multi-layer gain, which averaged only about 0.4 points, justifies the added complexity. Since the budget allows it, I'm considering a scheme that brings an early-layer mean into the final CLS token without disturbing the other tokens, since layer 0's CLS input is identical across every sentence.

I'm working out a rank-1 query update at layer 0 that would flatten the CLS attention while leaving other tokens alone. The catch is finding a direction u that is zero on every non-CLS token, since the shared β term complicates separating CLS from the rest.

I'm working out whether a least-squares choice of u can keep the other tokens' query shifts small, since CLS may sit along low-variance directions of the token second-moment matrix. The residual perturbation for non-CLS tokens depends on how distinct the CLS input is from the typical token span.

I'm weighing a way to make CLS attend uniformly at layer 0 with a rank-1 query update, but the simpler options I'm sketching — extra heads or key-side tricks — don't hold up. I'm now trying to reframe it around the residual stream rather than forcing CLS's attention to be uniform.

I'm considering making all twelve heads at layer 0 uniform for CLS by zeroing its query, so its attention output becomes a linear function of the mean of the token inputs, which should be full-rank enough to carry that information forward through the residual stream.

I'm weighing whether the carry-over of early-layer information survives the later BERT layers, since CLS's state keeps changing and would garble a mean0 signal. Protecting it in a subspace that the later layers barely touch is an option, but for roughly +0.4% it looks like a deep rabbit hole. I'm leaning toward a cheaper alternative with a similar expected gain.

I'm now considering whether pooling only the final layer's tokens while excluding CLS and SEP would help more, by down-weighting those special tokens through the key projection, though I'm unsure how much that would gain.

I'm now thinking about whether to first check how much CLS information survives the later BERT layers, by reading the CLS hidden state at each layer, before committing to the layer-0 uniform-attention scheme.

I'm considering skipping layers 5 through 10 entirely, by zeroing their attention and FFN contributions so the final hidden state comes straight from the earlier layers, which would let the early-layer lexical information reach the output without being garbled.

I'm weighing whether the mean-of-normalized-h5 pooling is worth it, given it scored lower than mean11. Combining the features would mean making CLS carry mean_L information, but the complexity is high for a gain of about 0.4% that may be noise. Still, the budget allows one trial of the layer-0 CLS-uniform approach, so I'm planning to try it while keeping checkpoint B as a fallback.

I'm weighing whether the CLS attention patterns in the middle layers would shift once CLS's key changes, since other tokens may rely on it as a no-op sink. That could disturb mean11 and erase the gain. I'm also noting that adding CLS11 earlier gave no improvement, which suggests the base model's CLS state contributes little on its own.

I'm weighing whether to drop the multi-layer idea, since BERT's later layers would likely garble any early-layer signal carried in CLS. The gain would probably come in well under the 0.4% I'd hoped for. I'm also checking the solution for robustness: sentence lengths stay under the 64-token limit, and the large Gamma scale in W2 should stay numerically manageable.

I'm checking whether the other layers' pooling choices could match or beat layer 11, including layer 9, which scores close to it. I'm also considering a two-stage kernel built on the layer 10 and layer 11 feed-forward outputs.

I'm dropping the lexical-keyword ensemble, since the FFN at layer 11 only sees CLS as a mean and can't pick out individual tokens. Instead I'll try whitening the features with PCA on the top-k components and see whether a quick cross-validation shows any gain.



The multi-layer gain is only +0.4% (97.7 → 98.1%), and adding base cls11 gave nothing, so I'm dropping that route. Next I'll test PCA-whitened spaces with fewer dimensions for the centre units.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy'); E=np.load('feats/edA_cls12.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef mkfolds(seed):\n    rng=np.random.default_rng(seed); fo=np.zeros(len(lab),int)\n    for c in range(8):\n        idx=np.where(lab==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%5\n    return fo\nFO=[np.load('feats/folds.npy'),mkfolds(7),mkfolds(8)]\ndef run(k,lams,a=2,t=0,rho=1.0,white=True,shrink=0.0):\n    res={l:[] for l in lams}\n    for fo in FO:\n        FS=[]\n        for f in range(5):\n            tr=fo!=f; mu=E[tr].mean(0); Xc=E-mu\n            U,s,Vt=np.linalg.svd(Xc[tr],full_matrices=False)\n            sv=s[:k]/np.sqrt(tr.sum())\n            Z=Xc@Vt[:k].T/((sv**2+shrink*np.mean(sv**2))**0.5 if white else 1)\n            R=np.sqrt(k); C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True)\n            FS.append(np.concatenate([gelu(a*(Z@C.T)/R+t), rho*Z/R],1))\n        for lam in lams:\n            pr=np.zeros(len(lab),int)\n            for f in range(5):\n                tr=fo!=f; te=~tr; F=FS[f]; Ftr=F[tr]; m=Ftr.mean(0); Y=np.eye(8)[lab[tr]]-1/8\n                Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n                pr[te]=((F[te]-m)@Wt).argmax(1)\n            res[lam].append((pr==lab).mean())\n    return {l:round(np.mean(v),4) for l,v in res.items()}\nfor k in [64,128,256]:\n    print('white',k, run(k,[0.3,1]))\nfor k in [128,256]:\n    print('white shrink1',k, run(k,[0.3,1],shrink=1.0))\n"
}
```


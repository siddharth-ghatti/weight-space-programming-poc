## reasoning summary

I'm comparing kernel accuracy across layers, where the jump from layer 8 to layer 9 stands out. The 98.3% figure for M9 would need pooling over layers 0 through 8, and I'm working out the indexing to confirm which layer's output that pooling covers.

I'm checking whether pooling M9 really needs k=10 layers, and whether the L1 kernel (k=2) at 97.5% fits the budget. Comparing the parameter costs for k=1, 2, and 3 with and without the 3072-wide term, the gap between 97.5% and 98.3% CV looks meaningful, though the test-set size is unknown and could explain some of it. I'm also weighing whether combining kernels across layers, or using a single k=1 layer with reassigned weights, would be cheaper.

I'm weighing a single-layer model against the 2-layer and 3-layer options, since attention could pool tokens in a way mean pooling can't. Building max-pool-like features would need token-level embeddings, which are awkward to get without reimplementing the forward pass.

I'm weighing whether a deeper model with heavily pruned FFNs could keep the contextual mixing that made the later layers stronger, since the cost of those layers scales with the FFN width. The open question is how much accuracy drops as the FFN shrinks, and I'd need to refit the kernel classifier on the pruned model's activations to find out.

I'm wondering whether pruning the FFNs heavily would still leave useful mean-pooled features, since the kernel classifier gets refit on the pruned model's activations anyway. Even noisy transformations seem to keep bag-of-words signal, so the real risk is that the deeper layers lose their contextual mixing and drift. I'm also considering dropping some middle layers, though that's still an open thought.

I'm weighing cheaper alternatives: a k=10 model with m=512 would come to about 56M parameters. I'm also comparing a k=2 model with m=3072 at roughly 38.6M, and a k=3 variant that scores lower on kernel accuracy, which makes M1 look stronger than the deeper features. Next I want to test whether combining the M0 and M1 features, or using multi-kernel sums or cosine-normalized kernels, does better.

I'm weighing whether 768 tanh pooler units could realize the kernel classifier, since kernel ridge over all 1496 training points isn't directly implementable. A Nyström approximation with 768 centers or random features might keep most of the accuracy, and I want to measure the drop. I'm also considering whether the pooling layer's FFN could feed GELU features back through the 768-dim residual stream.

I'm considering whether the pooler's 768 tanh units could act as random features in place of the full kernel, since kernel ridge over all 1496 points can't be realized directly. I want to measure how much accuracy that drops.

I'm checking whether the LN-normalized pooler input stays on a near-constant norm sphere, which would turn an RBF-style kernel into a dot-product one that tanh units can approximate.

I'm checking whether the pooler's LN output stays on a near-constant norm, and if the per-feature standardization can be folded into the pooler's linear map, so the tanh units can approximate the kernel's behavior on the pooled features.

I'm testing whether 768 tanh units with data-centered or random weights can stand in for the full kernel, using a centered Nyström-style setup and a tuned scale. I'm also considering Gaussian random features instead, and I'd like to compute the LayerNorm output offline to check how closely that simulates what the model will produce.

I'm checking the LN-normalized mean features on the kernels for layers 1 and 0, and I want to test 768 random tanh features as well.



RBF kernel accuracy rises only slightly with depth, reaching 98.3% at L9. Next I'll test whether layer combinations or kernel tweaks help the shallow layers, and check the loss from using 768 tanh units, then normalize the features per sample to match the pooler's LayerNorm input.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef tanh_rf_cv(X, y, nunits, scale, lams, kind='center', k=5, seed=0, rs=0):\n    n=len(y); S={l:np.zeros((n,8)) for l in lams}\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n        rng=np.random.default_rng(rs)\n        if kind=='center':\n            Cc=Z[rng.choice(tr,nunits,replace=False)]; \n            Hf=np.tanh(scale*Z@Cc.T)\n        else:\n            Wr=rng.standard_normal((X.shape[1],nunits)); bb=rng.uniform(-1,1,nunits)\n            Hf=np.tanh(scale*Z@Wr+bb)\n        Ht=Hf[tr]; hm=Ht.mean(0); T=np.eye(8)[y[tr]]; tm=T.mean(0)\n        A=Ht-hm; G=A.T@A\n        for l in lams:\n            Wt=np.linalg.solve(G+l*np.eye(nunits),A.T@(T-tm)); S[l][te]=(Hf-hm)@Wt+tm\n    return {l:round((S[l].argmax(1)==y).mean(),4) for l in lams}\nfor L in [0,1,9]:\n    X=LNr(M[L])\n    print(L,'lin LN', [round(cv_acc(X,y,l),4) for l in (100,300)])\n    for sc in [0.5,1,2,4]:\n        print(L,'center',sc,tanh_rf_cv(X,y,768,sc,(0.01,0.1,1)))\n    for sc in [0.5,1,2]:\n        print(L,'gauss',sc,tanh_rf_cv(X,y,768,sc,(0.01,0.1,1),kind='gauss'))\n"
}
```


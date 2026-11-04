## reasoning summary

The linear-only version with these folds reaches about 97.6%, better than the 97.46% from before, and the kernel should add roughly another third of a percent. Lower wlin looks better, so I'll try 0.1 and 0.0. The gain is moderate, around 0.35%. Before committing to this more complex design, I want to see whether other feature additions could help more.

I'm weighing whether other feature additions could beat the kernel route. M0 alone reached about 97.9% linearly, and I suspect combining it with the kernel could land near 98.2%, though M0 looks hard to deliver. I'm also considering attention-based pooling at layer 9, such as a head with a fixed query that acts like a soft max over a discriminative direction, but that seems complex. Shallow features from early and late layers have helped before, so that's worth keeping in view.

I'm now asking whether the earlier layers' token-level states could be recovered later, and concluding they can't through the residual stream alone. A cheaper route might be to pool at an earlier layer and carry that information forward, but keeping it intact through the intermediate layers is the obstacle. Another option is to stash the features in the CLS token so other tokens don't disturb them, though attention from other tokens in the first few layers could corrupt the stored values.

I'm working through whether the stash could be kept out of the other tokens' computation by zeroing its dimensions in the reading weights of layers 1 through 8, so only the LayerNorm statistics would be affected. That looks workable if the stash values stay small, but zeroing those columns would also strip out whatever original content those dimensions carry, which perturbs the network for every token.

I'm still weighing the stash idea: zeroing the stash dimensions in the reading weights of layers 1 through 8 would keep them out of other tokens' computation, but it would also remove whatever original content those dimensions carry. Leaving them readable is the alternative, though then writes from attention and the FFN could corrupt the stored values.

I'm planning to check the upper bound first by testing the kernel together with M0, assuming the stash works perfectly. I also want to see which shallow layer's mean helps most, M0, M1, or M2. Getting a stash at layer l would mean a uniform-attention head there writing into the stash dimensions, and since the heads are fixed, that would cost one original head unless I repurpose one, maybe at layer 0 or 1.

I'm weighing a cleaner way to get a stash at layer 0: a head that is already broad there could be made exactly uniform by zeroing its queries. The catch is that a head has only 64 value dimensions, all already in use, so I'd have to give up the least important ones to hold the stash.

I'm working out whether the CLS token could carry a stash without other tokens picking it up. The idea is to make the CLS query zero in that head while leaving other tokens' queries alone, which would take a rank-1 change to the query weights. Since the CLS embedding at layer 0 is identical for every input, that correction would be a fixed vector I can compute ahead of time.

I'm considering whether the CLS query could be altered so it attends uniformly while other tokens keep their original attention, using a rank-1 correction to the query weights at layer 0. Since the CLS input there is constant, the correction could be precomputed, with the main question being how much it leaks into the other tokens' embeddings.

I'm weighing whether the stash could go into the CLS token at layer 0 by using all twelve heads' uniform outputs, since each head's value projection only exposes a 64-dimensional slice of M0 but the output projection across heads spans the full width. Because the layer-0 CLS input is identical for every example, the required query correction could be computed in advance.

I'm considering whether the layer-0 tap could carry the stash through, since in the earlier tap experiment the information got overwritten by later layers. I'm not sure it would persist to layer 9, so I'll fall back to the variant that zeroes the stash dimensions in the reading weights instead.

I'm checking whether the LayerNorm scaling matters: each stash channel gets multiplied by a factor that depends on the per-layer standard deviations. Since the readout is a linear map over the CLS vector at layer 9, it already includes those stash dimensions, so a separate mechanism may not be needed, though the scaling factor varies from sentence to sentence.

I'm weighing how variable the scaling factor α is across sentences, and whether the offset term from mean subtraction matters. I can let the ridge regression absorb that offset as just another part of the layer-9 features, though I'd need to check whether it's correlated with anything else.

I'm wondering whether the LayerNorm scaling could be sidestepped, since the division in each LN makes the stash scale vary per sentence. Hoping the variance is small enough to ignore, I'm considering just testing it empirically: read the layer-9 features after the stash edit across all the sentences and see how much it actually hurts.

I'm weighing how much to test offline before running any more passes, since the budget is tight at around 3-5 passes for the stash experiment against 15 available. The plan is to first measure the upper bound with a kernel and a linear model on the combined CLS and M-features, including M0 as a perfect feature, and to see which of the early mean layers adds the most. I'm also considering whether a small number of stash dimensions, maybe 8 to 32 projected from M0, would capture most of the benefit.

I'm going to check the stash idea in a more controlled way: fit a combined ridge on the CLS, mean, and M0 features, then keep only the M0-derived part as an 8-dimensional stash. For an honest estimate, I'll fit that projection inside each cross-validation fold rather than on all the data. I also want to see how robust the result is when the stash is scaled by a random factor of roughly 10 to 20 percent, to mimic the LayerNorm variation.



The kernel readout beats linear by about 0.35% in repeated CV (roughly 97.95% vs 97.6%). Next I'll check offline whether shallow-layer mean features add anything before deciding whether the more complex stash mechanism is worth building.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nG={k:F[k].astype(np.float64) for k in F.files if k!='y'}\ndef stdz(X,tr): m=X[tr].mean(0); s=X[tr].std(0)+1e-6; return (X-m)/s\ndef ridge_pred(Xtr,Y,lam,Xte):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0)\n    n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return (Xte-xm)@Wt+ym, Wt\ndef rcv_lin(keys,lams,w=None):\n    w=w or [1]*len(keys); res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            X=np.concatenate([wi*stdz(G[k],tr) for k,wi in zip(keys,w)],1)\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams: res[l]+=(ridge_pred(X[tr],Y,l,X[te])[0].argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nprint('C9M9', rcv_lin(['C9','M9'],[300,1000]))\nfor k in ['M0','M1','M2','M3','M4','M6']:\n    print('C9M9+'+k, rcv_lin(['C9','M9',k],[300,1000,3000]))"
}
```


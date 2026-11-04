## reasoning summary

Two pooled layers already give 97.4% of the signal. Tracing the CLS token, layer 11 predicts layer 8 well (R² 0.80) but layer 4 only at 0.48, so about half the variance is lost over seven layers, though the discriminative part may survive better. The CLS states also look fairly stable from one sentence to the next.

The CLS states stay roughly constant across sentences, varying only about 14% in norm, so once the mean is removed the query residuals are small compared with typical token-state magnitudes. That should flatten the attention close to uniform, so the pooled vector sits near the layer-4 mean-pooled projection. Checking the token-level direction would need layer-4 token states, which I don't have yet.

I'm looking at whether injecting the mean later, around layer 6, keeps more of the CLS information, since the base layers 5–10 drop about half the variance. Spreading the injection across several layers might let the accumulated means stand in for the pooled signal, though later means could overwrite the earlier ones.

I'm now weighing whether summing the standardized means from layers 4 through 10 and concatenating M11 would work better than keeping them separate. The catch is that once means get injected into the CLS state at later layers, it starts to look like an average token, sitting in the middle of the token cloud, so the d-trick that relies on separating CLS from tokens would stop working.

The concern is that once means get injected into the CLS state at later layers, CLS starts to look like an average token, so the separating direction stops working.

The mean-injection idea at layer 6 looks worth checking: I'll compute the d-direction and the distribution of d·h for the layer-6 token states, then see how much the CLS query residual shrinks and how flat its attention gets.

I'm weighing whether to apply the d-trick only to some heads, since it changes CLS's queries and shifts tokens' attention to CLS only indirectly. That indirect effect looks small enough to accept, but injecting later layers loses more of the CLS information, with R² dropping to around 0.5–0.6.

I'm weighing which layer to tap the CLS state at, comparing pooled-mean combinations like [6, 11] and [8, 11] to see which keeps the most signal. The plan is to apply the d-trick at layer k so CLS attends nearly uniformly, leave layers k+1 through 10 unchanged, and then pool the layer-11 mean at the end.

I'm checking whether the CLS state can keep its distinctiveness if I split the final pooling: layer 11's CLS residual in a low-variance subspace, and the mean-pooled layer-11 signal in the complement. Earlier, plain superposition lost ground to simply concatenating the two, so separating the subspaces is the thing to test offline.

I'm now checking whether the CLS-tap design holds up offline, using the base layer-11 CLS states as a proxy. The idea is to split the final representation so the top-r subspace keeps the CLS residual while the complement carries the projected mean-pooled signal. I'll test that split against plain concatenation.

I'm thinking about using the layer-11 feed-forward network, but it only sees the already-superimposed state, so that won't separate the signals. A better option may be to split layer 11's heads: some pool the mean of the tokens, while others let the CLS state attend to itself and carry its own state through a separate channel.

I'm looking at making the CLS query attend to itself in some heads: scaling the d-direction so CLS's score with its own key is large and its score with token keys is near zero. Those heads would then carry CLS's own state through the value path, while the remaining heads pool the layer-11 token mean.

I'm planning the final CLS readout as a set of six heads that pool the layer-11 mean and six heads that copy the CLS state, with the output projection combining them into a concatenated representation. The feed-forward block would be zeroed out, so the final CLS state is roughly the normalized concatenation.

I'm checking how much accuracy is lost by compressing the layer-11 mean-pooled features down to 384 dimensions, using PCA on both that and the CLS states. The head-based readout depends on the CLS state staying distinct at layer 11, which looks likely, and the large attention scale should keep the self-attention weight near one even if the separation is imperfect.

I'm considering whether the CLS attention in the middle layers, k+1 through 10, could also be made to attend to itself, using a rank-1 addition along the d-direction, so that the token queries and keys barely shift. That would reduce how much of the mean-pooled information gets lost on the way to layer 11.

I'm weighing whether the CLS state can still be separated from the tokens if I give it self-attention in the middle layers, which would also shift how tokens attend to it. Choosing a direction that tokens' queries barely use might limit the disruption, but the tokens' queries span most of the head space, so the perturbation is hard to avoid. I'm leaning toward keeping the middle layers unchanged and measuring directly how much discriminative signal survives in the layer-11 CLS and mean states.

I'm going to compute the CLS and mean-token states for layers 5 through 9 and compare which tap layer pairs best with layer 11, since earlier layers bring more complementary signal but also more disruption.

I'm checking whether pooling tapped at layer k can be paired with the layer-11 mean by projecting each to 384 dimensions with PCA, to see how much is lost. I'm also considering tapping at several layers at once, though that would just superimpose the CLS state again. Keeping the design simple, the d-trick at layer k should leave CLS nearly uniform, since its residual query is small.

I'm wondering whether zeroing the CLS query's dependence on the variation direction would do even better, by cancelling it within the CLS subspace instead of only along one direction. That would perturb the token queries more, though, so the simple rank-1 version probably suffices.

I'm going to check how much discriminative signal the mean-pooled features carry at each candidate tap layer from 4 to 9, paired with the layer-11 mean. I also want to see how much is lost when each block is projected down to 384 dimensions with PCA, compared against the full concatenation.



A mid-depth tap works about as well as layer 0, and [CLS] states are nearly constant across sentences, so I'm checking a rank-1 correction to W_q that would make [CLS] attend uniformly while splitting layer 11's heads between pooling and self-copying, and I'll run more offline CV to choose the tap layer and confirm the 384+384 split costs little.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor k in [4,5,6,7,8,9]:\n    fn=lambda tr: np.concatenate([stdz(F[f'M{k}'],tr),stdz(F['M11'],tr)],1)\n    print('M%d+M11'%k,[cv(fn,l) for l in [30,100,200,400]])\ndef pca(X,tr,r,white=False):\n    Xs=stdz(X,tr); U,S,Vt=np.linalg.svd(Xs[tr],full_matrices=False); Z=Xs@Vt[:r].T\n    return Z/S[:r]*np.sqrt(tr.sum()) if white else Z\nfor r in [256,384,512]:\n    fn=lambda tr: np.concatenate([pca(F['M6'],tr,r),pca(F['M11'],tr,r)],1)\n    print('pca',r,'M6+M11',[cv(fn,l) for l in [30,100,200,400]])\nfn=lambda tr: pca(F['M11'],tr,384); print('M11 pca384 alone',[cv(fn,l) for l in [30,100,200]])"
}
```


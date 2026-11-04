## reasoning summary

The linear baseline sits at 97.64, and adding the stash brings it to roughly 98.0. Kernel alone reaches 98.02, and kernel plus stash hits 98.17, so the kernel gives about +0.38 while the stash adds only about +0.15 on top of it. The stash is more complicated and risky since it perturbs layers 0-8, so that small gain may not be worth it.

Now I'm working out the kernel readout design at layer 9, with the attention output and the layers after it treated as identity, so the input to the first LayerNorm is the CLS feature plus the mean-pooled term.

Next I'm checking whether the compressed inputs fit the 768-dimensional layer, since the full 1536-dimensional concatenation won't fit. I'm testing whether keeping the top principal components of each feature, so that the mean-pooled information lands in the low-variance directions of the CLS features, retains enough signal.

I'm now weighing whether the bottom principal components of C9 have enough room to absorb the projected M9 signal, since the identity residual means the CLS features can't be projected away. If the scaling of the mean-pooled term is large relative to that low-variance subspace, it could swamp whatever discriminative content those directions held, though that content was small to begin with.

I'm checking whether the low-variance directions of the CLS features can carry the mean-pooled signal, and whether standardization versus per-PC scaling changes how much those directions get amplified. Next I'm sketching an offline test: adding a scaled projection of M9 into the bottom principal components of C9, normalizing with LayerNorm, and seeing whether the kernel features still work once the normalization is applied.

I'm now comparing two kernel variants offline: cosine similarity on the raw layer-normalized features, and a dot-product kernel on the gamma-scaled features normalized by each unit's own norm, since the FFN can't divide by the sample norm directly.

I'm checking whether standardizing the features directly inside the FFN, with the first-layer weights applying the scaling, would avoid the outlier dimensions in C9 dominating the cosine similarity. The idea is that the outlier dimensions could be damped without changing the identity residual.

I'm checking whether the norm of the scaled input stays roughly constant across samples, since that decides whether the kernel can be approximated as a cosine. I'm also weighing the alternative of setting the first LayerNorm's gain to standardize the features directly, which would make the linear readout read the standardized input.

I'm checking that the layer-10 and layer-11 identity steps leave the output unchanged, since LayerNorm with unit gain and zero bias should be idempotent on an already-normalized vector.

Now I'm checking whether the output scale stays unsaturated: the tanh pooler's input depends on how large the FFN contribution is relative to the normalized residual, and I want to pick the constant so the argmax is preserved without pushing tanh into its flat region.

I'm weighing the trade-off between one unit per practice example and a multi-scale setup with two units per center at different β and θ, and I'm also counting the total units against the 3072 cap.

Now I'm checking whether reading the CLS state at layer 10 gives back x itself. Since the next layer's LayerNorm re-normalizes the input, if γ1 isn't all ones, the value read there is a normalized version of x rather than x, so I need to account for that.

I'm trying to read x exactly, but the read only gives back the normalized version, so I'm considering keeping γ1 at 1 with the channel dimensions zeroed, which leaves LN2(x) close to x without being exact. I'm also checking whether there's a way to avoid this, since the layer outputs won't expose the pre-LayerNorm values directly.

I'm checking whether the pooler can read the channel dimensions directly: the FFN writes into those dims, so the pooler's reading includes a known term from the x channel values divided by the FFN scale. That means the fit should subtract that term, which would require adjusting the B rows that touch the channel dimensions.

I'm working out how to absorb the leftover channel-dimension term x_ch[k]/c2 into the FFN head, so the pooler reading stays exact instead of only approximately negligible at a large c2.

I'm settling on accepting the small x_ch[k]/c2 leftover rather than trying to cancel it exactly, since with a large c2 it stays tiny next to the logit gaps.

I'm checking the next piece of the design: the FFN writes into the channel dimensions, so the pooler's reading picks up a term from them as well. Keeping the layer-norm gains at 1 and the bias at 0 and using a large scale constant should keep that leftover small enough to ignore, and I still need to confirm float32 precision holds.

I'm settling the mean-pooled input as u = c + G(M9 − m̄), with the mean folded into the bias, and taking G from the scaled top principal components of M9. I'm checking that the layer-9 output reads back x exactly, since with unit gain and zero bias the next LayerNorm leaves it unchanged.

I'm now setting up a design sweep over the number of principal components and the scale of the perturbation relative to the typical per-dimension spread of the CLS features. The plan is to check linear cross-validated accuracy on the constructed layer-normalized features and kernel accuracy on the mean-pooled targets, comparing the variants offline before fitting on the actual read-out.

I'm checking the typical per-dimension spread of the layer-9 features to set the perturbation scale, then running a linear cross-validation on the standardized features and a kernel cross-validation to compare the variants.



Offline results put kernel readout near 98%, and stashing M0 lifts it to about 98.17%, robust to 20% scale noise. Next I'll design the compression matrix G that maps M9's top whitened PCs into C9's low-variance directions, then check how much linear and kernel accuracy survives it.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef stdz(X,tr): m=X[tr].mean(0); s=X[tr].std(0)+1e-6; return (X-m)/s\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nprint('C9 per-dim std pct', np.percentile(C9.std(0),[1,50,99]), 'max', C9.std(0).max(), 'abs mean max', np.abs(C9.mean(0)).max())\nprint('C9 norm', np.percentile(np.linalg.norm(C9,axis=1),[1,50,99]))\ncC=np.cov(C9.T); ec,Vc=np.linalg.eigh(cC)   # ascending\ncM=np.cov(M9.T); em,Vm=np.linalg.eigh(cM); em=em[::-1]; Vm=Vm[:,::-1]\nprint('C9 eig asc', ec[:5], ec[380], ec[-5:]); print('M9 eig desc', em[:5], em[255], em[383])\ndef make_x(k,lam):\n    Q=Vc[:,:k]; P=Vm[:,:k]; D=1/np.sqrt(em[:k])\n    u=C9+lam*((M9-M9.mean(0))@P*D)@Q.T\n    n=(u-u.mean(1,keepdims=True)); n/=n.std(1,keepdims=True); return n\ndef cv(X,lams,kern=None):\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            Xs=stdz(X,tr)\n            if kern:\n                b,t,wl=kern; Xn=Xs/np.linalg.norm(Xs,axis=1,keepdims=True)\n                Z=np.concatenate([wl*Xs,gelu(b*(Xn@Xn[tr].T)-t)],1)\n            else: Z=Xs\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nfor k in [192,384]:\n    for lam in [0.1,0.3,1.0]:\n        X=make_x(k,lam)\n        print(k,lam,'lin',cv(X,[100,300,1000]))"
}
```


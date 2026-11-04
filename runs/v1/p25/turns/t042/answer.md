## reasoning summary

I'm checking that the layer-10 LayerNorm with gamma=1 and beta=0 is harmless, since it sits on an input that's already normalized and so passes through unchanged. Next I'm turning to the pooler's tanh and how one of its weight-row sums is set up.

I'm confirming that the final LayerNorm rescales all logits by a positive factor and subtracts a shared mean, so argmax is preserved. The pooler's channel term picks up a constant offset that doesn't change the ranking either. Next I'm weighing whether the expected gain from Plan N, roughly 97.4 to 97.86 CV, justifies the added complexity given the remaining run budget.

I'm weighing whether the extra complexity of Plan N is worth it at roughly 97.4 to 97.86 CV, given the budget left. Other options could give a bigger gain, like carrying the embedding mean M0 into the readout, but getting it through the residual stream without the later layers disturbing it looks tricky.

I'm checking how the residual stream carries a channel signal through each LayerNorm and attention/FFN update, since the signal gets rescaled and picks up noise at every layer. A strong signal would also change how the CLS token behaves downstream, while a weak one gets drowned out. Since the readout only needs the layer-9 CLS features, I'm looking at whether the weights can be adjusted to handle this.

I'm weighing whether the channel signal could be kept out of the other layers by zeroing the weights that read from or write to those dims, though the LayerNorm mean and std would still couple it to the rest of the CLS vector.

I'm weighing the channel-dimension approach against the LayerNorm effects: a small channel signal would barely shift the normalization stats, but zeroing the weights that read those dims would discard whatever they already carry. Choosing low-variance, low-importance dims seems like the way to keep that tradeoff manageable, though the gain of roughly 97.4 to 97.9 is still uncertain.

I'm considering whether the M0 embedding-mean feature could be carried into the layer-9 readout by stashing it in low-variance channel dims at layer 0, though the later layers could disturb it. A simpler route might give a comparable gain, so I'm weighing that against the added complexity.

I'm looking at stashing the M0 projection in channel dims that every layer ignores: zeroing the reads and writes for those dims and setting their LayerNorm gamma to 1 and beta to 0. The remaining concern is the per-token mean subtraction in each LayerNorm, which would still shift those values by a token-dependent amount.

I'm working out how the mean subtraction in each LayerNorm leaves the stashed channel values as a common positive scaling plus a shared offset, which would make the M0 contribution depend on how large the per-sentence scale turns out to be.

I'm sketching a way to stash the M0 projection in low-variance channel dims, but the per-sentence scale α from the LayerNorm chain would vary, so I'm wondering whether that variation could be measured or bounded.

I'm asking how large the per-sentence scale α from the LayerNorm chain actually varies, since it would change how much weight the stashed M0 channel gets relative to the other readout terms.

I'm looking at whether a constant reference channel, written at layer 0 and scaled the same way, could let me recover the per-sentence scale α from the differences between channels, since the common offset β would cancel out.

I'm weighing whether to give each channel its own reference constant so the per-sentence scale α could be recovered from their differences. But since LayerNorm can only divide over the whole vector, not selectively, that doesn't yield a usable correction, so I'm looking at adjusting the channel gammas to hold α near 1 on average and accepting the leftover variance.

I'm weighing a simpler route to the M0 gain, and I'm going to measure what's achievable across feature sets and nonlinear readouts with repeated CV before committing to Plan N. My CV estimates are noisy at roughly ±0.4%, but a gain that holds consistently across folds is probably real.

I'm sketching Plan N in more detail: a uniform attention head at layer 9 would carry the M9 signal through a value subspace, so the layer-9 input (the CLS features C9) and the M9 term get combined before the feed-forward block.

I'm sketching the FFN side of Plan N: the layer-9 input features go through LayerNorm, and since the normalized vector has a fixed length, a row of the first FFN weight matrix acting on a normalized center behaves like a cosine similarity. A GELU over that similarity, shifted by a threshold, would give kernel-like units I can then map back to the channel dims.

I'm planning to read the layer-9 features exactly by zeroing the layer-9 FFN and reading the layer-10 CLS input, since LayerNorm with gamma 1 and beta 0 leaves x unchanged. That gives me the exact normalized vector to fit a ridge model on, alongside the kernel units computed with numpy.

I'm considering whether the training points themselves should serve as the kernel centers, since with about 1,500 of them and 3,072 units that's enough capacity. Kernel ridge with those centers is the standard setup, so I'll go with it and then check the result with a real predict pass.

I'm considering whether the kernel features could be built on standardized inputs: since LayerNorm gamma can rescale each dimension of C9 per-dimension, I might get a whitened-like geometry that makes the cosine-GELU kernel behave better than on the raw mixed vector.

I'm working out how the LayerNorm gamma could rescale the combined vector per dimension, so I can place the M-principal components into low-variance channel dims. Since the first FFN matrix acts on the normalized features, I can fold any linear transform into it, which means the kernel could be computed in a transformed space rather than the raw one.

I'm weighing whether the kernel features should be computed in a transformed space, since the norm of Tx isn't fixed the way the norm of x is. A dot-product form like GELU of a scaled cosine-style term could stand in for the RBF, and I can compare such feature maps offline with the ridge fit. The final LayerNorm will also rescale the FFN output relative to the residual, so I need to keep that in view.

I'm checking that the final LayerNorm keeps the argmax unchanged: the logits reduce to one shared scale and a constant offset per class, so ranking survives. The catch is that the FFN must not write into the keep dimensions, or the linear part would be disturbed.

I'm checking that zeroing the layer-9 FFN and reading the CLS input at layer 10 gives me the exact normalized vector, and I'm thinking about how to set LN1's gamma and beta on the channel dims so their values stay at zero and the FFN output can be read off cleanly in the pooler.

I'm checking whether the tanh could saturate, since if the scaled logits get large the pooled outputs pin at ±1 and ties would break the argmax. The rescaling by the final LayerNorm depends on how big c2 is relative to the norm of f, so I need to keep the scaled logits in a moderate range, roughly within ±3.

I'm wondering why the constant channel needs to be large at all: the final CLS is LayerNorm of the residual plus a scaled channel term, so the question is whether that scale is actually doing work or just stacking an unhelpful contribution on top of the original low-variance values.

I'm settling the layer-9 attention so the CLS token averages over all tokens uniformly, which means zeroing the query weights and bias so the softmax stays flat. The pad mask still applies, and then I'll set the layer-1 norm's gamma to zero on the channel dimensions so the FFN can write into them without disturbing the readout.

I'm working out the layer-9 attention as a general linear map: with uniform attention, the CLS output is just a 768×768 transform of M9, so the pre-norm vector becomes C9 plus that transformed M9 plus a constant offset. Then LayerNorm with its gain and bias gives the layer-9 feature vector I'll be fitting on.

I'm thinking about whether a centered slice of M9's top principal components can be routed into the low-variance directions of C9. Putting the M9 signal there with a scaling factor should keep it separable from C9's dominant structure, though some of the low-variance C9 information gets mixed in.

I'm considering whether the low-variance directions of C9 would matter much for the ridge fit, since stronger regularization probably keeps the model from leaning on them. I could also try to cancel C9's contribution in the CLS residual by having one attention head focus only on the CLS token, though that would need a key direction that identifies CLS at layer 9.

I'm weighing whether to skip the extra pass: LN1 on the designed vector can be computed directly from the saved hidden states, and I'd only need two passes to confirm it matches the real forward pass.

I'm checking whether the sample-dependent scale σ(u) gets in the way, since the norm of C9 might vary a bit across inputs. Normalizing the scale is probably the natural choice for a cosine-style kernel, so I'll keep that in mind.

I'm going to run offline experiments first to pick the feature map, the number of principal components, and the kernel form, then compare the cross-validated accuracy against the linear baseline of 97.46%. If the gain holds at around 97.9%, I'll move ahead with it.

I'm thinking about normalizing the kernel inputs in a standardized space rather than the raw one, since BERT's hidden states have a few outlier dimensions with very large variance. Standardizing per dimension could help, and since the first FFN matrix can absorb any linear transform, the kernel can be formed on the transformed vector directly.

I'm picturing the kernel as a hinge-like function: GELU of a scaled cosine minus a threshold is near zero for most pairs and only fires for close neighbors, which could approximate the local behavior of an RBF kernel. Since the norm of the input is roughly constant, the RBF reduces to an exponential of the cosine, so I'll compare this shape against the real RBF offline.

I'm going to run the offline experiment now: loading the features, fitting a linear ridge baseline, and comparing kernel features built from GELU of scaled cosine similarity against the RBF version. After that I'll emulate the LayerNorm-rescaled input to see whether the gain holds.



A layer-9 linear readout reaches about 97.46% cross-validated accuracy. A kernel readout could reach roughly 97.9%, so I'm first checking offline whether that gain holds under realistic constraints.



## tool_use: run_python

```json
{
 "code": "import numpy as np, time\nF=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\nprint(folds.shape, folds[:10], np.bincount(folds) if folds.ndim==1 else None)\ndef gelu(z):\n    from scipy.special import erf\n    return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_fit(X,Y,lam):\n    xm=X.mean(0); Xc=X-xm; ym=Y.mean(0)\n    n,d=X.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\ndef cv(featfn, lams, fold=folds):\n    res={l:0 for l in lams}\n    for f in range(5):\n        tr=fold!=f; te=fold==f\n        Xtr,Xte=featfn(tr,te)\n        Y=np.eye(8)[y[tr]]-1/8\n        for l in lams:\n            Wt,b=ridge_fit(Xtr,Y,l); res[l]+=( (Xte@Wt+b).argmax(1)==y[te]).sum()\n    return {l:res[l]/len(y) for l in lams}\nZ=np.concatenate([C9,M9],1)\ndef lin(tr,te):\n    mu=Z[tr].mean(0); sd=Z[tr].std(0)+1e-6\n    return (Z[tr]-mu)/sd,(Z[te]-mu)/sd\nprint('linear', cv(lin,[100,300,1000]))\ndef kern(beta,theta,wlin=1.0):\n    def fn(tr,te):\n        mu=Z[tr].mean(0); sd=Z[tr].std(0)+1e-6\n        A=(Z[tr]-mu)/sd; Bt=(Z[te]-mu)/sd\n        An=A/np.linalg.norm(A,axis=1,keepdims=True); Bn=Bt/np.linalg.norm(Bt,axis=1,keepdims=True)\n        Ktr=gelu(beta*(An@An.T)-theta); Kte=gelu(beta*(Bn@An.T)-theta)\n        return np.concatenate([wlin*A,Ktr],1),np.concatenate([wlin*Bt,Kte],1)\n    return fn\nt=time.time()\nfor beta,theta in [(10,3),(10,5),(20,8),(5,1)]:\n    print(beta,theta, cv(kern(beta,theta,0.3),[1,10,100]), time.time()-t)"
}
```


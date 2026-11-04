## reasoning summary

I'm checking whether the shift δ lines up with the compensation direction: cosine 0.53, coefficient 0.38. That would mean part of the shift is in the compensation direction, so I'm asking whether it's overcompensating. I'm also working out what full overcompensation would look like if the true mean of the pruned contributions were zero, which would make the shift g·comp/σ_t.

I'm weighing whether the shift is worth chasing: the pre-LN std for layer 0 is unknown, so the proxy mean could be roughly twice the true one. Since the median z-score is small, the damage is mostly in the sample-varying part. I'm considering whether to test a larger removal size, but each variant costs a pass, and the size-accuracy tradeoff looks poor.

I'm comparing realization options for the head: an FFN acting as a kernel machine with GELU features, or pooler features at 768 centers. The FFN may be able to approximate an RBF-like kernel by using two GELU units per center, since the width allows it.

I'm weighing whether a two-unit GELU pair per center could approximate the RBF shape better than a single unit, though it might just overfit. I'm also checking how an exp-dot kernel on z-scored inputs differs from RBF, since the per-sample norm factor only rescales all class scores equally.

I'm checking whether the exp-dot kernel gives the same argmax as RBF once the per-sample and per-center factors are absorbed into the coefficients. The tanh pooler feature with a strongly negative bias approximates an exponential, so it could realize the same kernel, provided the pre-activation stays well below zero across the input range.

I'm checking whether the pooler's tanh feature can stand in for an exponential kernel: with a bias around -3 and a slope of 0.4, tanh stays in its saturated range and behaves like exp(0.8u) plus a constant. The float32 precision on the small variation around -1 should still be fine.

I'm checking whether a GELU-based FFN could realize an exp-like kernel the way the pooler's tanh can. GELU is Gaussian-shaped for large negative inputs and linear for large positive ones, so it doesn't give me an exponential. Routing the FFN output through a second stage would work in principle, but it gets complicated fast, so I'm looking at a different option where the FFN computes class scores directly.

I'm working out how to combine the two parts: the GELU features from the FFN pass through an 8-dim bottleneck into the pooler's decode, and I'd add exp features from the pooler at 760 centers. I'm checking whether a joint ridge over both feature sets, with logits as a linear function of each, can keep the tanh decode near-linear.

I'm checking whether the LN2 normalization and reserved-dimension scores would perturb the exp features, since the pooler's exp rows only see the non-reserved part of the normalized vector. I'm thinking about keeping the scores small so the perturbation stays negligible.

I'm checking whether a combined model could beat either approach alone: GELU features over all centers plus exp features at 760 centers, fit with one joint ridge. I'm also confirming the GELU hyperparameters are stable, since the two settings I tried give nearly the same accuracy.

I'm checking whether GELU can approximate exp(0.8u) over the range of cross-sample dot products, since an exact exponential would let the FFN realize the RBF kernel directly.

The GELU head with all centers looks like it lands near the kernel ridge result, and the pooler version is slightly lower. I'm considering whether to use both, and I'm noting that the overall accuracy to beat is the leave-one-out figure for RBF kernel ridge on the full data, which is about 98.

I'm comparing the accuracy estimates across validation schemes: 5-fold gives 98.2 for the m=3072 design, which I'd translate to roughly 98.6 on test, while LOO on the full data is 98.66. For the m=2560 variant, the subset LOO drop of about 0.48 suggests something near 98.1 to 98.2.

I'm weighing whether the 5-fold gap for the smaller design is large enough to matter, and whether it lands above or below the original 98.31 test figure. To settle it, I'm planning to run 5-fold and LOO on the same 1024-sample subset for both the base and the reduced model, across several hyperparameter settings, to get a steadier estimate of the drop.



About half the post-layer-0 shifts line up with the compensation direction, so the proxy means are somewhat off, though per-dimension shifts are small and the damage is mostly lost sample-varying signal. I'm now measuring the pruning loss on the 1024-id subset more robustly, across several hyperparameters with LOO and 5-fold×5 seeds, comparing base against pass_5.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nd=np.load('base_feats.npz'); y=d['y']; n=1024; ys=y[:n]\nFb=lnr(5*d['mean_hidden_L9'][:n]+d['cls_hidden_L9'][:n])\nFe=lnr(np.load('pass_5/cls_hidden_L10.npy'))\ndef cvk(F,y,gamma,lam,seeds=range(5)):\n    accs=[]\n    for s in seeds:\n        P=np.zeros((len(y),8))\n        for te in folds(len(y),5,s,y):\n            tr=np.setdiff1d(np.arange(len(y)),te)\n            Ztr,Zte=zs(F[tr],F[te]); T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n            al=np.linalg.solve(rbf(Ztr,Ztr,gamma)+lam*np.eye(len(tr)),T); P[te]=rbf(Zte,Ztr,gamma)@al\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nfor gm,lm in [(0.2,0.003),(0.4,0.003),(0.4,0.01),(0.8,0.01),(0.4,0.03)]:\n    print(gm,lm,'LOO base %.4f ed %.4f'%(krr_loo(Fb,ys,gm,lm),krr_loo(Fe,ys,gm,lm)), ' 5fold base %.4f ed %.4f'%(cvk(Fb,ys,gm,lm),cvk(Fe,ys,gm,lm)))\n"
}
```


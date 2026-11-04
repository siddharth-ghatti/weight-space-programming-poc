## reasoning summary

I'm comparing the exp(0.8u) kernel without the per-sample RBF normalization, which scores 98.6, a bit under the RBF's 98.70. The normalization still matters for the fit, since the regression uses the scaled kernel, even though predictions are invariant given the same α. I'm also checking the direct GELU(u) kernel with g=1 and b=0, which scores 98.6 with a minimum eigenvalue of 0.007, so it's positive semidefinite, and

Checking the remaining kernel options: GELU(0.5u) drops to 98.5, and any kernel with a nonzero offset goes non-PSD and breaks. The 2-unit combination lands at 98.56–98.6 but has erratic negative eigenvalues, so it's risky. The simplest robust choice is κ(u) = GELU(u) with one unit per center and λ around 0.003–0.01, which gives 98.6 under 20-fold CV.

Now I'm working out how the 1576 unused units are handled, and deciding that the scaling by the training-set mean and standard deviation still has to be applied to the kernel input.

I'm setting the FFN weights: the first layer's rows for the kept inputs get the z-scored center values divided by the standard deviation and by d, the residual inputs are zeroed, and each bias absorbs the mean-shift term. The second layer maps each unit's output to the residual channels using the scaled alpha coefficients, with the class-mean offset carried in the bias.

Still writing the output layer weights...

I'm settling on the scale and temperature for the final layer: the residual logit values stay small, so the per-sample norm shifts only about 0.07%, and a value of a=2 should keep the tanh pooling in its useful range.

I'm checking that the tanh pooling stays well within range: the centered scores stay in about [-0.5, 1], so the outputs remain in the ±0.96 band and the monotone tanh keeps the argmax intact. The leftover worry is whether the carried residual values, which feed into the LayerNorm statistics, shift things at test time, but they barely move μ and σ. Next I want to check the magnitude of the α coefficients at λ=0.003.

I'm simulating the head in float32 to check whether the argmax survives and how wide the margins are, and I'm also weighing the λ choice: 0.003 to 0.01 scores best, and larger λ keeps the α coefficients smaller and numerically safer. Before settling, I want to confirm the GELU kernel with 5-fold repeated across four seeds, since the RBF baseline there is 98.23.

I'm planning the verification pass: checking the logits for the 1024 edited ids against my numpy head output, where I expect agreement to about 1e-4. If that holds, the in-sample check is enough, and I'm weighing whether any of the remaining passes should go toward a larger pruned-layer experiment.

I'm weighing whether the m=2816 pruning experiment is worth running, since a drop of around 0.3 seems unlikely to clear a 0.15 LOO threshold. I'm also considering whether the verification pass could reveal a better pruning proxy than the energy-scaling estimate I have now.

I'm weighing whether to check the token_hidden outputs for all layers alongside the logits in one pass, since that request might exceed the 64-id limit. I'll keep the verification simple and build the head first before deciding on any further experiments.



The GELU kernel stays positive semidefinite and stable across λ with 98.6 accuracy, against 98.7 for RBF, while the b>0 kernels break down. I'm confirming it with 5-fold×4 seeds against RBF and checking coefficient sizes and float32 precision next.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\ny=np.load('base_feats.npz')['y']; X1=np.load('X1_full.npy'); F=X1[:,KEEP]\nres={}\nfor s in range(4):\n    for te in folds(len(y),5,s,y):\n        tr=np.setdiff1d(np.arange(len(y)),te)\n        Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]; T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n        for nm,kf in [('rbf',None),('gelu',lambda G: gelu(G))]:\n            if nm=='rbf': Ktr=rbf(Ztr,Ztr,0.4); Kte=rbf(Zte,Ztr,0.4)\n            else: Ktr=kf(Ztr@Ztr.T/dd); Kte=kf(Zte@Ztr.T/dd)\n            w,V=np.linalg.eigh(Ktr); VT=V.T@T\n            for l in [0.003,0.01]:\n                res.setdefault((nm,l),{}).setdefault(s,np.zeros((len(y),8)))[te]=Kte@(V@(VT/(w+l)[:,None]))\nfor k,v in res.items(): print(k, round(np.mean([(p.argmax(1)==y).mean() for p in v.values()]),4))\n# full fit coefficients\nZ,_=zs(F); dd=F.shape[1]; K=gelu(Z@Z.T/dd); T=np.eye(8)[y]; T=T-T.mean(0)\nfor l in [0.003,0.01]:\n    al=np.linalg.solve(K+l*np.eye(len(y)),T); print(l,'alpha absmax',np.abs(al).max().round(2),'sum|alpha|',np.abs(al).sum(0).max().round(1), 'train acc', ((K@al).argmax(1)==y).mean())\n"
}
```


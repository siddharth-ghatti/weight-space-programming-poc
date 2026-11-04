## reasoning summary

The plateau is broad, somewhere around 97.7 to 98.2 percent, so I don't need to chase the single best point. a=2, t=0, lam=0.5 to 1, rho=1 sits comfortably in the middle and should be robust. Now I'm wondering whether multi-layer features could push accuracy further, perhaps by building a kernel over concatenated feature sets.

I'm working out how to build the kernel machine, and I'm noting the final model should use all 1496 practice points as centers, which should do slightly better than cross-validation. I also need to check whether the harness uses exact GELU, though that difference probably won't matter. For layer 11, the input to the feed-forward network is essentially the normalized features E, and the first 1496 intermediate units hold kernel-based weights built from centered, normalized training features.

I'm setting up the final FFN weights: the first 1496 units carry the kernel-based pre-activations, the next 16 handle the linear part through paired GELU units, and the per-class offsets from centering have to go into the output bias, since they affect the argmax.

I'm checking how the readout scale interacts with the residual: with a large Γ, the score term dominates the leftover input component, so each class readout is driven mainly by its orthogonal direction.

I'm checking whether the large residual scale causes float32 trouble in the layer norm, and it looks fine, since the eps is tiny. I'm also considering whether to zero out the input contribution, but I'll keep the residual as is. For the pooler, I'm setting the first eight rows to the scaled class vectors with zero bias and leaving the rest at zero, so each pooled score is a tanh of the projected final representation.

I'm checking how the readout is set up: each pooled score is a tanh of the projected final representation, with the scale chosen so the tanh stays well away from saturation. Since the class vectors are zero-mean, the leftover input contributes little unless every score is near zero.

I'm sorting out the FFN bias: the intercepts go into b2 through the readout, and the 1496 kernel units plus 16 linear ones leave 1512 of the 3072 slots, so the rest can stay zeroed. I also want to confirm that LN2 on the already-normalized input returns roughly x, so the FFN input really does match E.

I'm checking a few more points around a=1.5 and t=0, varying lam between 0.1 and 0.5, to see whether the result holds up. I'll also rerun it on a different fold split to make sure the choice isn't just noise.



Several settings in the 97.7–98.2% CV band look equally good, so I'll recheck that region on a second, independent fold split before picking one.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy'); E=np.load('feats/edA_cls12.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef mkfolds(seed):\n    rng=np.random.default_rng(seed); fo=np.zeros(len(lab),int)\n    for c in range(8):\n        idx=np.where(lab==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%5\n    return fo\ndef run(fo,a,t,lams,rho=1.0):\n    out={}\n    FS=[]\n    for f in range(5):\n        tr=fo!=f; mu=E[tr].mean(0); sd=E[tr].std(0)+1e-6\n        Z=(E-mu)/sd; R=np.sqrt(768); C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True)\n        FS.append(np.concatenate([gelu(a*(Z@C.T)/R+t), rho*Z/R],1))\n    for lam in lams:\n        pr=np.zeros(len(lab),int)\n        for f in range(5):\n            tr=fo!=f; te=~tr; F=FS[f]; Ftr=F[tr]; m=Ftr.mean(0); Y=np.eye(8)[lab[tr]]-1/8\n            Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n            pr[te]=((F[te]-m)@Wt).argmax(1)\n        out[lam]=round((pr==lab).mean(),4)\n    return out\nfor seed in [7,8]:\n    fo=mkfolds(seed)\n    for a in [1.5,2]:\n        print(seed,a,run(fo,a,0,[0.1,0.3,0.5,1]))\n"
}
```


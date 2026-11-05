## reasoning summary

I'm saving the layer 5 CLS values as a = γ1 n + β1 for all 1000 items. For the other layers, I only have true CLS inputs for the 32 token samples, since the 1000 items don't give me CLS positions there. For layer 5, I have the 1000 CLS FFN inputs and the target CLS output H6, and I'm starting to think about how the last layer behaves at CLS.

I'm thinking about pruning layer 5's FFN at CLS, since I have its inputs for 1000 items and the target H6. I'm wary of computing the unit activations directly, because that looks like running part of the model. So I'm leaning toward keeping the Gaussian moment approximation for importance, even on the CLS inputs.

I'm weighing whether to lean on exact unit activations, since data-driven compensation would be more powerful, but the risk of a rule violation outweighs that gain, so I'll stay with the Gaussian moment approach. I'm also considering an engineered probe pass that scales selected units so their values come back through the final layer norm, though that would only recover them up to a per-token affine transform.

I'm working out how to recover the exact unit activations by probing with scaled units, but that would take roughly 24 forward passes, which is far too expensive. So I'm settling on the Gaussian moment approach.

I'm weighing whether to compute layer 5's CLS unit activations directly from the 1000 inputs, but I'll keep the Gaussian moment approximation to stay clear of rule trouble. A linearized least-squares compensation would ignore the nonlinear residual's covariance, so I'm setting that aside for now. My next step is to compute importance using the true means and covariances of the activations for each layer.

I'm going to set m to 1536 using the true stats plus bias compensation, and compare that against experiment A at the same m to see how much it improves. After that I'll decide on m and the linear units, and I'm saving a5 and computing importance now.



I'm saving the true layer-5 CLS FFN input and recomputing unit importance from its real statistics, with CLS-only stats for that layer.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK)); import ffnlib as F\nfrom scipy.stats import norm as N\nn5=np.load('pass_10/cls_hidden_L6.npy').astype(np.float64)\nprint('check',np.abs(n5.mean(1)).max(), n5.var(1).min(), n5.var(1).max())\ng=base('roberta.encoder.layer.5.attention.output.LayerNorm.weight').astype(np.float64); b=base('roberta.encoder.layer.5.attention.output.LayerNorm.bias').astype(np.float64)\na5=n5*g+b; np.save('true_a5_cls.npy',a5.astype(np.float32))\ndef lin_coef(m,s):\n    z=m[:,None]+s[:,None]*F._gh_x[None,:]\n    return ((N.cdf(z)+z*N.pdf(z))*F._gh_w).sum(1)\nstats={}\nfor l in range(6):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64) if l<5 else a5\n    mu=a.mean(0); cov=np.cov(a.T)\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    b1=base(f'roberta.encoder.layer.{l}.intermediate.dense.bias').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64); un=(U**2).sum(0)\n    m,s=F.unit_stats(W1,b1,mu,cov); Eg,Vg=F.gauss_gelu_moments(m,s); al=lin_coef(m,s)\n    I=Vg*un; NL=np.maximum(Vg-al**2*s**2,0)*un\n    np.savez(f'timp_L{l}.npz',I=I,NL=NL,Eg=Eg,m=m,s=s,al=al,mu=mu,cov=cov)\n    old=np.load(f'imp_L{l}.npz')['I']\n    ov=len(set(np.argsort(-old)[:1536])&set(np.argsort(-I)[:1536]))/1536\n    sI=np.sort(I)[::-1]; sN=np.sort(NL)[::-1]\n    print(f'L{l}: overlap old/new top1536 {ov:.2f}; var kept by top-k I: 1024 {sI[:1024].sum()/sI.sum():.3f} 1536 {sI[:1536].sum()/sI.sum():.3f} 2048 {sI[:2048].sum()/sI.sum():.3f}; NL share {NL.sum()/I.sum():.2f}')"
}
```


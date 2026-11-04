## reasoning summary

I'm checking whether the 12% drift after layer 0 is measured with the outlier dimensions included, since I only kept the non-outlier dims. I also want to see whether the hidden-state norm at layer 1 is dominated by a few outliers like dim 308, so I'll compute the drift per dimension to find which ones actually move.

I'm also wondering whether the drift measure is skewed by a shared offset in the mean, which a z-scoring classifier would ignore. I think a better check is the relative difference after centering each sample, so I can see how much the sample-specific variation changes.



The importance score looks like activation std times output-column norm under a Gaussian approximation. I'm checking whether the drift is a common shift that z-scoring ignores or a sample-varying distortion that hurts, before changing the method.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nd=np.load('base_feats.npz'); n=1024\nfor L in range(10):\n    a=np.load(f'pass_5/mean_hidden_L{L}.npy')[:,KEEP]; b=d[f'mean_hidden_L{L}'][:n][:,KEEP]\n    shift=np.linalg.norm(a.mean(0)-b.mean(0))/np.linalg.norm(b.mean(0)+1e-9)\n    ac=a-a.mean(0); bc=b-b.mean(0)\n    # per-dim z-scored\n    sd=bc.std(0)+1e-6\n    var_rel=np.linalg.norm((ac-bc)/sd)/np.linalg.norm(bc/sd)\n    corr=np.mean([np.corrcoef(ac[:,j],bc[:,j])[0,1] for j in range(0,760,7)])\n    worst=np.argsort(-np.abs(a.mean(0)-b.mean(0))/sd)[:5]\n    print(L, f'shift {shift:.3f} centered-zrel {var_rel:.3f} meancorr {corr:.3f}', KEEP[worst])\n"
}
```


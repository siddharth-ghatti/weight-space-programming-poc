## reasoning summary

I'm checking that the RES entries in layers 2 and up really were left alone, since their W2 rows are already zeroed out, so the compensation there comes to zero anyway. The compensation formula is now pinned down as the base plus the pruned columns of W2 times their mean. I'm also wondering whether the mean-based compensation is off, because the layer-1 output shifted by about 11% of the mean norm, and the means came from Gaussian proxies.

I'm weighing whether the mean-based compensation is even right, since the proxy means came from Gaussian approximations of the x stats that may be off. My plan is to check the observed layer-1 shift against the compensation vector to see whether it overshoots or is uncorrelated with it.



The compensation rule is now reconstructed as b2 += W2[:, pruned] · proxy_mean[pruned]. Next I'll regress the 11% mean shift after layer 0 on that compensation to see whether it over- or under-shoots.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nd=np.load('base_feats.npz'); n=1024\np=np.load('ffn_proxy.npz')\na=np.load('pass_5/mean_hidden_L1.npy'); b=d['mean_hidden_L1'][:n]\ndelta=(a-b).mean(0)\ncomp=W('bert.encoder.layer.0.output.dense.bias')-base('bert.encoder.layer.0.output.dense.bias')\ng=base('bert.encoder.layer.0.output.LayerNorm.weight')\n# post-LN effect of a pre-LN vector v: g*(v - mean(v))/sigma\nv=comp-comp.mean(); pred=g*v\nc=(delta@pred)/(pred@pred); print('coef', c, 'cos', delta@pred/np.linalg.norm(delta)/np.linalg.norm(pred))\nprint('norm delta', np.linalg.norm(delta), 'norm residual', np.linalg.norm(delta-c*pred))\nprint('top delta dims', np.argsort(-np.abs(delta))[:8], np.sort(np.abs(delta))[::-1][:8].round(3))\nprint('base mean norm', np.linalg.norm(b.mean(0)), 'sample sd typical', np.median(b.std(0)))\n# per-dim delta relative to sample sd\nz=delta/(b.std(0)+1e-6); print('z-shift top', np.sort(np.abs(z))[::-1][:8].round(2), 'median', np.median(np.abs(z)).round(3))\n"
}
```


## reasoning summary

The predictions match the expected values exactly in-sample, so the implementation itself checks out. The relres of 0.21 on the R decode is the open issue: the R coordinates may carry a common shift from the prepared model, since I mean-centered both rc and sc over the 8 coordinates. I'm now checking whether the carried signal is the sum of the contributions, which would explain why the R-coordinate pattern isn't exactly proportional.

The per-layer contributions don't share one scale, since each passes through a different number of LayerNorms. So a single scale fits poorly: the fitted values run from 0.97 to 5.19. I'm considering estimating a scale for each layer and rebalancing the gain to match.

I'm estimating a separate scale for each layer's contribution with per-sentence least squares, then checking the mean and spread per layer. The LayerNorm chain explains the mismatch: the layer-2 carry gets scaled much more by the time it reaches the output than the layer-10 carry does.

The per-layer scales look different: roughly 4.6 for the layer-2 carry, 3.3 for layer 6, and 2.4 for layer 10, since each passes through a different number of LayerNorms. My plan is to set each layer's gain inversely to its scale so the final contributions line up, then verify on a second pass and recompute the decode scale from the reference value.

I'm checking how the out-of-sample accuracy is affected by this scale mismatch, since the in-sample 100% is just overfitting. To get a fair estimate, I'll simulate held-out predictions with per-layer scale factors, starting by computing each sentence's per-layer values.



The network's predictions match the numpy expectation on every sentence, and the non-R features barely moved. The R decode isn't a single scale, so I'm estimating a separate scale for each carry layer and rebalancing the per-head gains.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab)\nF=dict(np.load(WORK/'prep_feats.npz')); J=np.load(WORK/'joint_v2.npz')\nc12=np.load(WORK/'v2_c12.npy')\nparts=['m2','m6','m10']\nS=[F[p]@J[p] for p in parts]            # [N,8] each\nSc=[s-s.mean(1,keepdims=True) for s in S]\nr=c12[:,R]; rc=r-r.mean(1,keepdims=True)\nA=np.zeros((N,3)); res=np.zeros(N)\nfor i in range(N):\n    X=np.stack([Sc[k][i] for k in range(3)],1)*0.1\n    coef,_,_,_=np.linalg.lstsq(X,rc[i],rcond=None); A[i]=coef\n    res[i]=np.linalg.norm(rc[i]-X@coef)/np.linalg.norm(rc[i])\nprint('per-layer scale mean', A.mean(0).round(3), 'std', A.std(0).round(3), 'relres', res.mean().round(4), res.max().round(4))\n# global (pooled) fit of per-layer scale\nX=np.concatenate([np.stack([Sc[k][i] for k in range(3)],1) for i in range(N)])*0.1\ncoef,_,_,_=np.linalg.lstsq(X,rc.reshape(-1),rcond=None); print('pooled per-layer scales', coef.round(3))\nnp.save(WORK/'v2_layer_scales.npy', A)\n"
}
```


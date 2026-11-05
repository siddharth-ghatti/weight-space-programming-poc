## reasoning summary

I'm comparing the experiment A selection errors, around 0.13 to 0.25 per layer, against plain top-1024 selection at roughly 0.125 to 0.17, which turn out to be similar. With linear units, m=1024 looks like it drops to about 0.07 to 0.15, and m=1536 lower still.

I'm comparing the error bounds for linear units at m=1536, which look roughly half of experiment A's. Summing per-layer errors, the m=1536 setup comes out near 2.5% disagreement, versus about 5% for m=1024. I'm also wondering whether the Gaussian bias terms could be systematically off for heavy-tailed units, since that might matter more than the random error.

I'm settling on the implementation: m-r NL-selected units plus r repurposed slots whose weights get overwritten with the linear-unit construction, with r chosen per layer between 32 and 128.

I'm working out the pruned-unit contribution: the linear part depends on the input through a matrix M built from the pruned units' weights and their Gaussian slope coefficients, and the constant part is absorbed into a bias compensation. Next I'll set up the low-rank approximation of that matrix.

I'm weighing the best rank-r approximation of M, measured in the covariance-weighted norm, since the whitened form could be ill-conditioned for the layer-5 covariance.

I'm settling on the low-rank step: the best rank-r approximation of M, measured in the covariance-weighted norm, comes from the top-r left singular vectors of M·S12. Each linear unit then gets an input vector built from M's transpose applied to its output direction.

I'm now choosing the constant C and scale ε so that each linear unit's pre-activation stays well in GELU's linear region, with ε·σ_k = 1 and C = 10, which keeps z roughly within 5 standard deviations of 10. At that point the GELU is effectively the identity, so the output reduces to the linear term plus a constant offset.

I'm checking the edge cases now: near z=0 the GELU matches the linear term exactly, but for strongly negative pre-activations it falls to zero and the error grows. Raising the offset C to around 15 keeps the bias cancellation numerically safe in float32, so I'll set ε·σ_k = 1 with C=15.

I'm now handling the bias compensation: the pruned units' mean contributions get folded into the output bias, minus the offset terms from each linear unit. For the kept set, I'll take the top m−r units by nonlinear importance and add r low-importance pruned units to be overwritten with the linear construction, then rebuild the per-layer keep lists from that union.

I'm settling on a sweep order: I'll first try m=1024 since it saves the most, and fall back to m=1536 if agreement drops too far. Within each layer I'll pick r from a small set of candidates, using the estimated error to decide.

I'm weighing whether the m=1024 setup with linear units lands near 95% agreement, since its summed error is well below experiment A's. I'm going to test m=1536 first to check that the approach holds up, predicting about 97-98% agreement, and then use the result to model how error maps to agreement.



Calibration puts m=1536 at roughly 0.33 total error and m=1024 at about 0.70, versus 1.13 for experiment A. I'll build m=1536 first, repurposing pruned slots as linear units per layer and folding the remaining means into the output bias, with r chosen per layer to minimize error.



## tool_use: run_python

```json
{
 "code": "code=r'''\nimport numpy as np\nRS=(0,16,32,48,64,96,128,160)\ndef plan_layer(l, m, base, rs=RS):\n    d=np.load(f'timp_L{l}.npz'); I,NL,al,cov,Eg,mu=d['I'],d['NL'],d['al'],d['cov'],d['Eg'],d['mu']\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n    ev,evec=np.linalg.eigh(cov); S12=evec*np.sqrt(np.maximum(ev,0))\n    best=None\n    order=np.argsort(-NL)\n    for r in rs:\n        if r>m: continue\n        keep=order[:m-r]; pr=np.setdiff1d(np.arange(3072),keep)\n        M=(U[:,pr]*al[pr])@W1[pr]\n        Ul,sv,_=np.linalg.svd(M@S12)\n        e=(NL[pr].sum()+(sv[r:]**2).sum())/I.sum()\n        if best is None or e<best[0]: best=(e,r,keep,pr,M,Ul[:,:r],sv[:r])\n    return best\ndef build_layer(l, m, base, C=15.0, rs=RS):\n    e,r,keep,pr,M,Ur,sv=plan_layer(l,m,base,rs)\n    d=np.load(f'timp_L{l}.npz'); Eg,mu=d['Eg'],d['mu']\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\n    U=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\n    b2n=b2+U[:,pr]@Eg[pr]\n    # repurpose r pruned slots (lowest importance pruned ids)\n    I=d['I']; slots=pr[np.argsort(I[pr])[:r]]\n    W1n=W1.copy(); b1n=b1.copy(); Un=U.copy()\n    for k in range(r):\n        u=Ur[:,k]; v=M.T@u; s=sv[k]          # std of v.(a-mu) is s\n        eps=1.0/max(s,1e-8)\n        j=slots[k]\n        W1n[j]=eps*v; b1n[j]=C-eps*(v@mu)\n        Un[:,j]=u/eps\n        b2n-= (C/eps)*u\n    kept=np.sort(np.concatenate([keep,slots]))\n    assert len(kept)==m and len(set(kept))==m\n    return dict(err=e,r=r,kept=kept,W1=W1n,b1=b1n,U=Un,b2=b2n)\ndef apply_all(m, base, save_edit, layers=range(6), C=15.0, rs=RS):\n    keep={}; info=[]\n    for l in layers:\n        res=build_layer(l,m,base,C,rs)\n        pre=f'roberta.encoder.layer.{l}.'\n        save_edit(pre+'intermediate.dense.weight',res['W1'].astype(np.float32))\n        save_edit(pre+'intermediate.dense.bias',res['b1'].astype(np.float32))\n        save_edit(pre+'output.dense.weight',res['U'].astype(np.float32))\n        save_edit(pre+'output.dense.bias',res['b2'].astype(np.float32))\n        keep[l]=res['kept'].tolist(); info.append((l,res['r'],round(res['err'],4)))\n    return keep, info\n'''\nopen(WORK/'lu.py','w').write(code)\nimport sys; sys.path.insert(0,str(WORK)); from lu import *\nfor e in edits(): drop_edit(e)\nkeep,info=apply_all(1536,base,save_edit)\nprint(info)\nprint(set_structure(layers=list(range(6)),ffn_keep=keep))"
}
```


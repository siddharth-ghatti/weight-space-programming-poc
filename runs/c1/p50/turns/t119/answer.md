## reasoning summary

The rank-56 truncation keeps almost all of the head's output energy, and the regression fit suggests compensation would only recover a sliver of the already tiny residual, so it's not worth chasing. For the provisional carry weights, I'm leaning toward a joint fit on the base M9 model with 1496 parameters rather than the pass_19 M9e setup, since the final weights get refit regardless.

Next I'm defining the carry values as a scaled linear part of the fit, normalizing the features by their root-mean-square norm and fixing the scale so each dimension has roughly unit spread. After that I'll work out the value and output projections for head 0 of layer 0 from the truncated SVD, splitting the singular-value scaling between them.

I'm working out how to offset the residual from the truncation: the approximation error is (M − M̂)m, and since the SVD was on centered data, the mean term M·μ needs its own correction, which I'm folding into the output bias for the kept dimensions. That makes the output M·m minus the approximation error term plus a constant. Next I'm setting up the carry dimensions 56 to 63, filling the kept entries from the scaled carry weights and the residual entries from the removed dimensions.

I'm setting up the carry dimensions 56 through 63 now: each gets a value-weight row that picks up the scaled carry weights on the kept entries and a −1 on its residual entry, plus a bias that offsets the mean. On the output side, the matching residual rows read straight from those carry columns, and the output bias for the residual dimensions stays at zero.

Now I'm zeroing out the query rows and biases for head 0, which makes its logits flat regardless of the key. Layer 0's attention and output layer norms keep the residual gains at one and offsets at zero, while the FFN's residual columns and rows get cut.

Now I'm setting the layer 1 projections, output bias and FFN rows for the residual dimensions, and the layer 2 and layer 6 heads are being checked to confirm they write nothing into the kept dimensions.

I'm setting up the exposure for each layer in sequence: the layers before it keep their chosen 1024-unit FFN sets, and the layer being exposed gets its output zeroed along with its attention and output layer norms reset to identity.

I'm setting up a helper module that works out the intended layer-norm values for each layer, since the expose step overwrites them. For each layer being exposed, the attention and output norms get identity settings while the output projection is zeroed.

I'm now defining the drift-corrected fit for each layer: the attention outputs go through the intended layer-norm settings, the FFN contribution is added back, and I'm computing the per-token mean and spread across all 768 dimensions, including the residual ones.

I'm settling the drift-correction target for the KEEP dimensions, scaling the hidden-state targets and subtracting the mean and the existing offsets, while the residual dimensions are set to zero. I'm also choosing how to select the FFN energy and ridge-refit the kept columns against the base weights, and picking the mixing weight between the replication and drift targets per layer on a held-out split.

I'm writing the helper module lib4.py to compute the intended layer-norm values and the drift-corrected fit for each layer, then applying the layer 0 and 1 surgery to get provisional weights.



I'm writing `lib4.py` next, with the layer-0/1 surgery, the provisional carry weights, and the exposure helpers for the drift-corrected re-prune.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, json\nRES=np.array([83,88,222,254,444,495,506,601]); KEEP=np.setdiff1d(np.arange(768),RES)\nEXPO_IDS=json.load(open('expo_ids.json'))\ndef pre(l): return f'bert.encoder.layer.{l}.'\ndef design_ln(l, base):\n    P=pre(l); out={}\n    for nm in ['attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','output.LayerNorm.weight','output.LayerNorm.bias']:\n        v=base(P+nm).astype(np.float32).copy()\n        if l in (0,1): v[RES]=1.0 if nm.endswith('weight') else 0.0\n        out[nm]=v\n    return out\ndef regcomp(Xk, Xr, lam=1.0):\n    mk=Xk.mean(0); mr=Xr.mean(0); A=Xk-mk; B=Xr-mr\n    Bc=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]), A.T@B)\n    return Bc, mr-mk@Bc     # Xr ~ Xk@Bc + c\ndef cut_reads(Wm, bm, Bc, c):\n    # Wm: [out,768] reading x; replace x_res by Xk@Bc+c\n    Wn=Wm.astype(np.float64).copy(); bn=bm.astype(np.float64).copy()\n    Wn[:,KEEP]+=Wm[:,RES]@Bc.T; bn+=Wm[:,RES]@c; Wn[:,RES]=0\n    return Wn.astype(np.float32), bn.astype(np.float32)\ndef expose(l, save_edit, set_structure, keeps, m):\n    P=pre(l)\n    save_edit(P+'attention.output.LayerNorm.weight', np.ones(768,np.float32))\n    save_edit(P+'attention.output.LayerNorm.bias', np.zeros(768,np.float32))\n    save_edit(P+'output.dense.weight', np.zeros((768,3072),np.float32))\n    save_edit(P+'output.dense.bias', np.zeros(768,np.float32))\n    save_edit(P+'output.LayerNorm.weight', np.ones(768,np.float32))\n    save_edit(P+'output.LayerNorm.bias', np.zeros(768,np.float32))\n    fk={i:keeps[i] for i in range(l)}; fk[l]=list(range(m))\n    return set_structure(layers=list(range(l+1)), ffn_keep=fk)\ndef restore_ln(l, save_edit, base):\n    for nm,v in design_ln(l,base).items(): save_edit(pre(l)+nm, v)\n'''\nopen('lib4.py','w').write(code); print('written')"
}
```


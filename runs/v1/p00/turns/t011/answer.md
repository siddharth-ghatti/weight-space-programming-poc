## reasoning summary

I'm dropping IDF weighting since it downweights function words like "not" and "must" that matter here, so a uniform mean stays as the simpler choice. I'm also checking whether [CLS] and [SEP] belong in the features; they're constant, so including them is harmless. For the ridge penalty I'd pick λ with repeated CV across several seeds over 0.2 to 0.4, and I'm leaning toward 0.3.

Now I'm working out the edit math: the ridge weights on the projected features, and how the value rows and bias collapse into one linear map from the token features to the eight scores.

I'm picking the residual dimensions for the CLS output: the ones with the smallest combined gamma and beta magnitudes, leaving out known outlier dimensions. I'm also checking the bound on each component, which depends on the normalized value and the bias.

I'm estimating how large the scale s needs to be: with s around 1e4, the perturbation stays far below the margins, so its effect is negligible. I'm also checking float32 precision in the layer-norm step, since the pre-norm entries stay around a few thousand and the variance remains well-behaved. The remaining question is whether the ridge outputs for a sentence sum to zero, which the centered targets should guarantee.

I'm checking that the ridge outputs for each sentence sum to zero, which follows from the centered one-hot targets. I'm also confirming that after the scaled perturbation the layer norm leaves the CLS vector as expected, and I'm considering a small pooler scale around 0.05 to see how the class projections come out.

Next I'm checking the pooler scale: the output should stay monotone in the class scores. A small scale around 0.02 keeps the nonlinearity mild, and the classifier weights can stay at 1 since argmax is unaffected.

I'm confirming that BERT's pooler output feeds the classifier, and that the zeroed attention and FFN outputs in the later layers leave nothing to disturb the CLS path. I'm also checking that the layer norm epsilon is 1e-12, and whether the harness applies the saved edits on top of the base model.

I'm checking whether the edits replace the saved tensors outright rather than adding to the base weights, which seems to be what the harness means by base plus saved edits. I'm also considering whether attention might be uniform, since with zeroed query and key projections every score would come out equal.

I'm now setting up the sanity checks: I'll confirm the value-projection algebra by feeding random inputs through the value rows and bias and comparing against the direct ridge-weight computation.

I'm picking the 16 residual dimensions with the smallest bias and gamma magnitudes, skipping the known outliers, and I'll write the build script to a file so the whole construction can be reproduced.



IDF weighting hurt because function words like "not" and "must" carry class signal, so the uniform mean stays. I'm now writing the edits into build.py along with self-checks on random vectors, not real activations.



## tool_use: run_python

```json
{
 "code": "build = r'''\nexec(open(WORK/'feat_lib.py').read())\nK_POS = 30; LAM = 0.3; S_BIG = 1e4; EPS_POOL = 0.02\ndef fit_ridge0(X, y, lam):\n    Yt = np.eye(8)[y]; Yt = Yt - Yt.mean(0)\n    return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@Yt)\nPm = build_P(K_POS)\nFa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\nWm = fit_ridge0(Fa, y, LAM)                 # 768 x 8, z = Wm^T Fa\nA = (Pm @ Wm).T / sc                         # 8 x 768 acting on n_j (normalized embedding)\ng = base('bert.embeddings.LayerNorm.weight').astype(np.float64); bt = base('bert.embeddings.LayerNorm.bias').astype(np.float64)\nG = A / g[None,:]                            # acts on x0 = g*n + bt\ngbias = -(A @ (bt/g))\n# readout directions q_c = (e_a - e_b)/sqrt2 on dims with small |beta|\norder = np.argsort(np.abs(bt) + 0.1*np.abs(g))\ndims = order[:16]\nQd = np.zeros((768,8))\nfor c in range(8):\n    Qd[dims[2*c],c] = 1/np.sqrt(2); Qd[dims[2*c+1],c] = -1/np.sqrt(2)\nf32 = lambda a: np.asarray(a, dtype=np.float32)\nz768 = np.zeros((768,768)); \n# ---- layer 0 attention\nL0 = 'bert.encoder.layer.0.'\nsave_edit(L0+'attention.self.query.weight', f32(z768)); save_edit(L0+'attention.self.query.bias', f32(np.zeros(768)))\nsave_edit(L0+'attention.self.key.weight', f32(z768)); save_edit(L0+'attention.self.key.bias', f32(np.zeros(768)))\nWv = np.zeros((768,768)); Wv[:8] = G; bv = np.zeros(768); bv[:8] = gbias\nsave_edit(L0+'attention.self.value.weight', f32(Wv)); save_edit(L0+'attention.self.value.bias', f32(bv))\nWo = np.zeros((768,768)); Wo[:, :8] = S_BIG*Qd\nsave_edit(L0+'attention.output.dense.weight', f32(Wo)); save_edit(L0+'attention.output.dense.bias', f32(np.zeros(768)))\n# ---- all layers: LN -> identity affine, FFN output -> 0; layers 1..11 attention output -> 0\nfor l in range(12):\n    p = f'bert.encoder.layer.{l}.'\n    for ln in ['attention.output.LayerNorm.','output.LayerNorm.']:\n        save_edit(p+ln+'weight', f32(np.ones(768))); save_edit(p+ln+'bias', f32(np.zeros(768)))\n    save_edit(p+'output.dense.weight', f32(np.zeros((768,3072)))); save_edit(p+'output.dense.bias', f32(np.zeros(768)))\n    if l>0:\n        save_edit(p+'attention.output.dense.weight', f32(z768)); save_edit(p+'attention.output.dense.bias', f32(np.zeros(768)))\n# ---- pooler & classifier\nWp = np.zeros((768,768)); Wp[:8] = EPS_POOL*Qd.T\nsave_edit('bert.pooler.dense.weight', f32(Wp)); save_edit('bert.pooler.dense.bias', f32(np.zeros(768)))\nWc = np.zeros((8,768)); Wc[np.arange(8),np.arange(8)] = 1.0\nsave_edit('classifier.weight', f32(Wc)); save_edit('classifier.bias', f32(np.zeros(8)))\nnp.save(WORK/'readout_dims.npy', dims)\nprint('saved; sc', sc, 'dims', dims)\n'''\nopen(WORK/'build.py','w').write(build)\nexec(build)\n# --- algebra self-checks (on random vectors, not activations)\nrng = np.random.RandomState(1); u = rng.randn(768)\nx = g*u + bt\nctx = W('bert.encoder.layer.0.attention.self.value.weight').astype(np.float64)[:8]@x + W('bert.encoder.layer.0.attention.self.value.bias')[:8]\nprint('cancel check', np.abs(ctx - A@u).max(), np.abs(A@u).max())\nprint('A kills ones/type0/pos?', np.abs(A@np.ones(768)).max(), np.abs(A@rm1(T0)).max(), np.abs(A@Pe[:64].T).max(), 'vs word', np.abs(A@Wd[2000:2100].T).max())\nprint('Q zero-mean', np.abs(Qd.sum(0)).max(), 'orthonormal', np.abs(Qd.T@Qd-np.eye(8)).max())\nprint('dims beta/gamma', bt[dims].round(3), g[dims].round(3))\nprint(len(edits()))\n"
}
```


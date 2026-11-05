## reasoning summary

I'm weighing the held-out result against the in-sample bound: 0.170 is above the 0.129 ceiling, and more data might bring it near 0.13-0.14, which is a modest gain at best. I'm also wondering whether the short paragraphs are simply noisier, since the probe used 120-330 token paragraphs. Now I want to focus on the CLS token in particular.

I'm considering whether the CLS position deserves special treatment, since the final output depends on it and it may carry attention-sink statistics that differ from other tokens. Collecting CLS inputs at each of the first five layers would cost about five more probe passes, and I'm weighing whether a per-layer sensitivity check could tell me where the error actually comes from first.

I'm working out the budget first: about 40 passes, so roughly six for CLS data, six for sequential correction, and ten for evaluation. Then I'm considering a sequential refit, where each layer's kept W2/b2 are fit so the pre-LN2 output matches the base, which would also partly absorb upstream error.

I'm now checking whether refitting the attention output projection could correct the error too, but I can't observe the per-head context vectors directly, so that path is messy and I'll set it aside for now.

I'm planning a sensitivity check to locate the noise: one experiment prunes only layers 0-4 to m=1024 while keeping layer 5 full, and another prunes only layer 5, both with least-squares compensation and zeroed dropped columns at m=3072. Comparing those against an all-layers m=1024 run should show which part of the stack is responsible.

I'm going to run the diagnostics at m=1024 before committing to more CLS data collection: one pass pruning only layers 0-4 with layer 5 kept full, and one pruning only layer 5. Those two results should show whether the last layer is driving most of the error, and I can combine them to approximate the all-layers case.

I'm now worried that the layer-5 compensation is being evaluated on the same CLS inputs it was fit to, so its error would look smaller than it really is. A split would give a fairer estimate, fitting on one half of the examples and testing on the other.

I'm splitting the examples in half, fitting the layer-5 compensation on the even-indexed ones and evaluating on the odd ones, so the noise estimate isn't inflated by in-sample fit. The 32 probe paragraphs are small enough that I can just leave them out of the evaluation for layers 0-4.

I'm going to adjust the builder so it can run in zeroing mode and fit the layer-5 compensation on only a subset of the examples.



Even an in-sample fit only lifts the m=1024 layer from 0.195 to 0.129 (0.170 held out), and QR selection does no better than variance ranking. Next I'll run a sensitivity split that zeroes dropped units' outputs to simulate per-layer widths, fitting any layer-5 compensation only on even-indexed paragraphs.



## tool_use: run_python

```json
{
 "code": "builder2 = r'''\ndef build_layer_zero(Lr, m, lam_rel, fit_mask=None):\n    \"\"\"Prune by zeroing dropped W2 columns (structure stays full). fit_mask: rows of data used for fitting.\"\"\"\n    pre = f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\"); b2 = base(pre+\"output.dense.bias\")\n    A = ffn_data(Lr)\n    if fit_mask is not None: A = A[fit_mask]\n    H = gelu(A@W1.T + b1)\n    sc = H.var(0)*(W2**2).sum(0)\n    keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\n    Hk = H[:,keep]; mk = Hk.mean(0); T = H[:,drop]@W2[:,drop].T; mt = T.mean(0)\n    if lam_rel is None: B = np.zeros((m,768))\n    else:\n        Xk = Hk-mk; G = Xk.T@Xk; lam = lam_rel*np.trace(G)/m\n        B = np.linalg.solve(G+lam*np.eye(m), Xk.T@(T-mt))\n    W2n = np.zeros_like(W2); W2n[:,keep] = W2[:,keep] + B.T\n    save_edit(pre+\"output.dense.weight\", W2n.astype(np.float32))\n    save_edit(pre+\"output.dense.bias\", (b2 + mt - mk@B).astype(np.float32))\n    return keep.tolist()\n'''\nopen(WORK/'builder.py','a').write(builder2)\nexec(open(WORK/'builder.py').read())\nfor e in edits(): drop_edit(e)\nclear_structure()\nfor Lr in range(5): build_layer_zero(Lr, 1024, 1.0)\nprint(edits(), structure()['m'])"
}
```


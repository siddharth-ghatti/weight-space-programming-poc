## reasoning summary

Layer 5 gains the most from least-squares compensation because its CLS-only data sits on a low-dimensional manifold, while layers 1 and 3 only improve modestly. I'm also weighing the sample count: with m=1024 above the 750 training samples, the ridge term matters.

Now I'm weighing whether to test m=1536 across all six layers with variance ranking and mean compensation, then m=1024, since each test costs one pass. The catch is that m has to be shared across layers, so the most sensitive layer sets the limit, and I'm considering whether to drop layer 5 and refit the head instead.

I'm weighing which mix of pruned layers and width gets closest to a 55M budget, and I'd need a test pass for each option. To judge fidelity I'll check agreement with both the original head and a refit head. Since in-sample agreement may flatter the result, I'm planning to also look at the low-margin examples, which resemble the test set more closely, before running the first experiment with all six layers at m=1536.

I'm going to run the first experiment with layers 0–4 using least-squares compensation at lam_rel=1, and layer 5 at 0.1, since that's where the gains were measured on held-out paragraphs. The mid-layer improvement at m=1536 is modest, but it's a real held-out gain, so it's worth keeping. Next I need to derive the new down-projection and bias so the dropped contribution is absorbed into W2 and b2.

Next I'm writing a builder function that takes the chosen width, the layers to prune, and per-layer regularization, and saves the edited weights and kept-unit structure together. The dropped columns of W2 will just be zeroed, since they're sliced away anyway.



LS compensation cuts relative error modestly in layers 0-4 and strongly in layer 5 (0.256 to 0.109), so next I'm building a reusable builder and testing all six layers at m=1536 for the real effect on predictions.



## tool_use: run_python

```json
{
 "code": "builder = r'''\nfrom scipy.special import erf\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef ffn_data(Lr):\n    if Lr<5: return np.load(WORK/f'ffn_in_L{Lr}.npy')\n    return np.load(WORK/'ffn_in_L5_cls.npy')\ndef build_layer(Lr, m, lam_rel, extra_A=None, score='var'):\n    pre = f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\"); b2 = base(pre+\"output.dense.bias\")\n    A = ffn_data(Lr) if extra_A is None else extra_A\n    H = gelu(A@W1.T + b1)\n    sc = H.var(0)*(W2**2).sum(0)\n    keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\n    Hk = H[:,keep]; mk = Hk.mean(0); T = H[:,drop]@W2[:,drop].T; mt = T.mean(0)\n    if lam_rel is None:\n        B = np.zeros((m,768))\n    else:\n        Xk = Hk-mk; G = Xk.T@Xk; lam = lam_rel*np.trace(G)/m\n        B = np.linalg.solve(G+lam*np.eye(m), Xk.T@(T-mt))\n    W2n = np.zeros_like(W2); W2n[:,keep] = W2[:,keep] + B.T\n    b2n = b2 + mt - mk@B\n    save_edit(pre+\"output.dense.weight\", W2n.astype(np.float32))\n    save_edit(pre+\"output.dense.bias\", b2n.astype(np.float32))\n    return keep.tolist()\ndef build_all(layers, m, lam=None):\n    lam = lam or {}\n    ffn_keep = {}\n    for Lr in layers:\n        ffn_keep[Lr] = build_layer(Lr, m, lam.get(Lr, 1.0 if Lr<5 else 0.1))\n    return set_structure(layers=layers, ffn_keep=ffn_keep)\n'''\nopen(WORK/'builder.py','w').write(builder)\nexec(builder)\nprint(build_all([0,1,2,3,4,5], 1536))\ns = structure(); print(s['layers'], s['m'], len(edits()))"
}
```


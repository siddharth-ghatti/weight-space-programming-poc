## reasoning summary

Linear augmentation doesn't help: adding full-rank features makes error worse, which is odd, maybe overfitting or the regularization being uneven across the blocks. Either way, the linear part is already captured and isn't a lever here. The compensation looks close to its limit for this data and architecture, so the remaining knob is m. I'll evaluate m=1536 on the new data with a forward pass.

I'm refitting layers 0-4 at m=1536 on all the data with lam 0.05. Before that, I need to restore the layer 5 probe edits, then fit layer 5 using the 2000 CLS samples with lam 0.1. I'll also read the logits and the layer 6 CLS hidden states, since that's cheap, and possibly test m=1280 afterward.



The linear augmentation adds nothing, so the compensation is near its limit and m is the remaining knob. I'm rebuilding every layer at m=1536 with the larger dataset, then checking with a forward pass.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\ncode5 = r'''\ndef fit_L5(m, lam_rel, save=True):\n    pre=\"roberta.encoder.layer.5.\"\n    W1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64); b2 = base(pre+\"output.dense.bias\")\n    A = np.r_[np.load(WORK/'ffn_ed_L5_cls.npy'), np.load(WORK/'ffn_in_L5_cls.npy')].astype(np.float64)\n    H = gelu(A@W1.T+b1)\n    sc = H.var(0)*(W2**2).sum(0); keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\n    Hk=H[:,keep]; mk=Hk.mean(0); T=H[:,drop]@W2[:,drop].T; mt=T.mean(0); X=Hk-mk; G=X.T@X\n    B=np.linalg.solve(G+lam_rel*np.trace(G)/m*np.eye(m), X.T@(T-mt))\n    if save:\n        W2n = W2.copy(); W2n[:,keep] += B.T; W2n[:,drop] = 0\n        save_edit(pre+\"output.dense.weight\", W2n.astype(np.float32))\n        save_edit(pre+\"output.dense.bias\", (b2 + mt - mk@B).astype(np.float32))\n        ks = load_keeps(); ks[5] = keep.tolist(); save_keeps(ks)\n    return keep\ndef build_m(m, lam_tok=0.05, lam5=0.1):\n    for L in range(5):\n        fit_layer(L, m, lam_tok)\n    fit_L5(m, lam5)\n    ks = load_keeps()\n    return set_structure(layers=list(range(6)), ffn_keep={l: ks[l] for l in range(6)})\n'''\nwith open('seq.py','a') as f: f.write(code5)\nexec(code5)\nrestore_layer(5)\nimport shutil; shutil.copy(WORK/'keep_sets.json', WORK/'keep_sets_m1024.json')\nprint(build_m(1536))\nprint(edits())"
}
```


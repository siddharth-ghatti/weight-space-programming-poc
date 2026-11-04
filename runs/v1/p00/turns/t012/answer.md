## reasoning summary

I'm checking the cancellation test: the context weights in float32 against A times u, and the discrepancy comes out to about 7.9e-8, which looks fine. The per-position magnitudes of A·p_j top out around 0.0104, far below the word-level values, since A acts on the normalized quantities n_j = e/σ, scaling things up by roughly 20. I already covered that with the leakage test, which flipped 1 of 1496 cases. Next I'm turning to the precision question for the value weights G = A/γ.

Now I'm checking the δ residual on the chosen dimensions, since the layer-0 CLS values scale with γ and the normalized inputs, which puts δ near 2 and δ/s around 2e-4, so that looks fine. I also want to confirm the edited weights are consistent: the shapes, the full edit list, and the analytic prediction of the expected argmax against the ridge accuracy.

I'm checking the layout of the value vectors: HF's self-attention reshapes the 768-dim value output into 12 heads of 64, so context dims 0–7 map back to head 0's first eight dims.

I'm checking that the head-0 value dimensions map back correctly through the output projection and layer norm, and that the edit list and shapes are complete. I'll also record the analytic predictions for documentation.

I'm confirming the LABELS list is in the same order as the task's label order, since the class index depends on it.



The algebra checks all pass, with the readout ignoring nuisance directions and staying zero-mean and orthonormal. Next I'm confirming label order, listing the saved edits, and recording the analytic expectations in notes.md.



## tool_use: run_python

```json
{
 "code": "print(LABELS)\nE = edits(); print(len(E)); \nnames = sorted(E) if isinstance(E,(list,dict)) else E\nprint([n for n in names if 'layer.0.' in n or 'pooler' in n or 'classifier' in n])\ncat = {c['name']:c for c in catalog()}\nfor n in names:\n    a = W(n); assert tuple(a.shape)==tuple(cat[n]['shape']) and np.isfinite(a).all()\nprint('all shapes ok')\nnotes_add = \"\"\"\n## Run 12: BUILT & SAVED (build.py, feat_lib.py in working folder)\n- K_POS=30 positional PCs (pos 0..127) + ones + type0 projected out; ridge, no intercept, lambda=0.3 on proxy features\n  F = P * mean_j(word_emb[t_j]/sigma~_j)/sc, sigma~^2=(||c(w+type0)||^2+||c(pos)||^2)/768, sc=5.157.\n- Proxy CV (5-fold): ~95.9% acc / macro-F1 ~0.959. Robust to sigma model (approx/word/none all ~same), no-intercept => scale-invariant.\n- Positional leakage with K=30: 1/1496 train-pred flips (K=0 gives 45 flips -> projection essential).\n- IDF-weighted pooling tested: hurts (function words like 'not','means','must' matter). Uniform mean kept.\n- S_BIG=1e4, EPS_POOL=0.02, readout dims (pairs) saved in readout_dims.npy. 106 edited tensors.\n- Algebra checks passed: gamma/beta cancellation exact (8e-8), A\u00b7ones=0, A\u00b7type0=0, A\u00b7pos ~0.01 vs words ~0.3.\n\"\"\"\nopen(WORK/'notes.md','a').write(notes_add)\n"
}
```


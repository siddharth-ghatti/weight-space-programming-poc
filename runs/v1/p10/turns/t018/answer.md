## reasoning summary

I'm double-checking the final layer, where the key weights stay unchanged and the zero query gives scores of zero plus the mask, which looks right. I'm also verifying the layer 0 bias on D, where the FFN output reduces to s times the difference between A transpose x and the center, and that checks out too. I'm now looking at the stolen neurons in layer 0, whose W_1 reads x across all dimensions while A[D] is zero.

I'm planning a pass over 1024 ids to read the edited network's cls and mean hidden states at every layer, which should show the magnitudes of the D dimensions per layer and the final CLS features. I'm also weighing a no-carry baseline, since the non-D dimensions should be close to the mean of the edited network's layer 11 output, with the base m11 as the comparison.



## message

Before spending the pass, I'll record the v1 design in notes.md.

## tool_use: run_python

```json
{
 "code": "open('notes.md','a').write(\"\"\"\n## Findings (base reads: pass_1 = junk 100 ids L12 cls; pass_2 = base cls+mean all layers ids p0000-p1023)\n- Ridge CV (5-fold, 1024): mean_hidden >> cls. mean L8-L11 ~95.6-96%. mean L0 ~93.8%. concat(mean L0, mean L11) ~97.4%.\n- 8 directions of L0 (from joint ridge on [m11,m0], lam=0.4) stacked w/ m11 -> ~97%.\n## Design v1 (builder.py: build(g_attn,g_out,s,B,c,A,D,center))\n- D dims=[222,601,495,88,254,444,233,42] carry s*A^T x (layer0 FFN, 8 stolen low-norm neurons, gelu linear regime B=10).\n- Layers 1-10: D protected (reads/writes zeroed, LN gamma_D=g, beta_D=0).\n- Layer 11: Q=0 uniform attn, V=I, Wo=1000I, FFN out zero, LN gamma1 beta0 -> CLS = LN(mean tokens h11).\n- Files: A_proj.npy, D_dims.npy, center.npy, helpers.py, builder.py\n- Plan: pass3 = read edited 1024 (cls+mean all layers) -> check D magnitudes & CV; recalibrate g; fit pooler/classifier (argmax-preserving tanh).\n\"\"\")\n"
}
```


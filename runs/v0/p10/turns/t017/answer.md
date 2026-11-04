## reasoning summary

The deep pairs come out around 0.965 against 0.96 for single pooling, about half a percent better, though some of that may just be the doubled feature count acting as regularization. Small gain overall. I'm leaning toward v1 as single pooling at layer 8 with the pass-through, and fitting from there. I'm also weighing whether running CLS pooling across several late layers could recover the concatenation benefit.

I'm still weighing whether pooling CLS across several late layers, with the CLS accumulating means, could recover the concatenation benefit. The v1 design needs more thought, especially the cost of CLS-only attention at layer 8 given how much the CLS state varies there.

I'm weighing whether the CLS-only uniform attention at layer 8 is worth the cost, since making the CLS query uniform would require projecting out its variation. Another option would be a large key component or lowering the temperature, but that gets fiddly for roughly half a percent of gain. I'll skip it, go with v1, and put the remaining effort into validation and tuning the pooling layer and regularization on the full data.

Sticking with the single-layer mean pooling, I'm weighing whether to mix CLS and mean heads at the pooling layer or keep every head as a plain mean. Keeping it simple seems right, and I'm also checking whether to exclude CLS and SEP from the mean, though the CV setup already included them. The remaining layers should pass the signal through unchanged, since each LayerNorm is effectively idempotent here.

I'm weighing the output scale of the pooler: with the weight set to s·I and no bias, large outlier dimensions in the normalized layer-8 mean could push tanh into a noticeably nonlinear range. A smaller scale like s = 0.02 would keep it near-linear, at the cost of larger classifier weights, which isn't a real problem numerically.

I'm considering standardizing the pooler weights so the pooled features come out roughly unit-scaled and tanh stays near-linear across all dimensions, which would help the ridge fit. I can't compute the exact layer-8 normalized statistics here, so I'd estimate the scale from the activations instead, which should be good enough for choosing the pooler scaling.

I'm setting the pooler's bias so the pooled features come out roughly standardized, with a small scale of about 0.1 so tanh stays nearly linear. I can estimate the per-dimension spread from the mean activations and a typical per-sample scale, without reimplementing the forward pass, since the classifier is fit on the actual pooled features anyway.

Next I'll read the pooled features for all 1496 items in two passes, then fit a ridge classifier with cross-validation and save the weights. After that I'll run a verification pass to confirm the implementation matches my computed logits, keeping in mind that predictions on the fitting data will look optimistic.

Next I'm setting up the pooling layer's attention so every head is uniform: zeroing the query weights and biases, and making the value projection the identity. The output projection will be a large scaled identity, so the layer effectively returns the mean of the hidden states, though the CLS state still leaks in a little.

Setting up the pooling layer's remaining weights: the output projection gets a scale of 1000, the output LayerNorm stays at identity, and the feed-forward block is zeroed out. The three layers after it get their attention and feed-forward outputs zeroed too, so they just pass the signal through.

I'm settling the pooler's scaling: I'll set its weight to the scale divided by the per-dimension spread and its bias to match, so the pooled features come out standardized. The classifier stays at zero until I fit it. I'm also weighing whether layer 8 or layer 9 is the better pooling point, since their cross-validation scores are within noise and I don't have full-data reads to separate them yet.

I'm settling on layer 8 for the pooling point, since its cross-validation scores edge out layer 9's even if the gap is small. I also need to write up the setup notes before I start editing the weights.



I'm skipping the deeper layer pairs, since they add only about 0.5% and need fiddly attention edits, and building a simpler v1 instead: layer 8 pools, layers 9–11 pass through, and the pooler becomes a near-linear standardiser. Then I'll fit the head in closed form on the real pooled outputs.



## tool_use: run_python

```json
{
 "code": "\nopen('notes.md','a').write(\"\"\"\n## Findings (passes 1-3)\n- pass_1 wasted (1 id). pass_2: base cls/mean hidden all layers for p0000-p1023. pass_3: base token_hidden+attention, 64 texts (every 16th id).\n- mean_hidden includes CLS and SEP (exact uniform mean over valid tokens).\n- Ridge CV (std features): mean L8 ~0.963, L9 ~0.96, L11 0.954, cls L12 0.93. concat(mean L0, mean L9) ~0.971 but needs 256+ dims of L0 info -> not injectable cheaply. Deep pairs (a,11) ~0.965.\n- Tokens attend heavily to CLS in layers 2-4 (sink), little after layer 5. CLS state ~constant in layers 1-3.\n## v1 plan\n- layer 8: Q=0 (uniform attn), V=I, O=1000 I, LNs=(1,0), FFN out=0 -> CLS = LN(mean h8). Layers 9-11 pass-through (attn out=0, ffn out=0, LN=(1,0)).\n- pooler: diag(s/sd), bias -s*mu/sd (near-linear tanh). Classifier fit by ridge on read pooled outputs.\n\"\"\")\nI = np.eye(768, dtype=np.float32); Z768 = np.zeros(768, np.float32)\na = 8\npre = f'bert.encoder.layer.{a}.'\nsave_edit(pre+'attention.self.query.weight', np.zeros((768,768),np.float32))\nsave_edit(pre+'attention.self.query.bias', Z768)\nsave_edit(pre+'attention.self.value.weight', I)\nsave_edit(pre+'attention.self.value.bias', Z768)\nsave_edit(pre+'attention.output.dense.weight', 1000*I)\nsave_edit(pre+'attention.output.dense.bias', Z768)\nsave_edit(pre+'attention.output.LayerNorm.weight', np.ones(768,np.float32))\nsave_edit(pre+'attention.output.LayerNorm.bias', Z768)\nsave_edit(pre+'output.dense.weight', np.zeros((768,3072),np.float32))\nsave_edit(pre+'output.dense.bias', Z768)\nsave_edit(pre+'output.LayerNorm.weight', np.ones(768,np.float32))\nsave_edit(pre+'output.LayerNorm.bias', Z768)\nfor l in range(a+1, 12):\n    pre = f'bert.encoder.layer.{l}.'\n    save_edit(pre+'attention.output.dense.weight', np.zeros((768,768),np.float32))\n    save_edit(pre+'attention.output.dense.bias', Z768)\n    save_edit(pre+'attention.output.LayerNorm.weight', np.ones(768,np.float32))\n    save_edit(pre+'attention.output.LayerNorm.bias', Z768)\n    save_edit(pre+'output.dense.weight', np.zeros((768,3072),np.float32))\n    save_edit(pre+'output.dense.bias', Z768)\n    save_edit(pre+'output.LayerNorm.weight', np.ones(768,np.float32))\n    save_edit(pre+'output.LayerNorm.bias', Z768)\n# pooler standardizer: approximate LN(mean h8) stats\nM8 = np.load('pass_2/mean_hidden_L8.npy').astype(np.float64)\nZn = (M8 - M8.mean(1,keepdims=True))/M8.std(1,keepdims=True)\nmu = Zn.mean(0); sd = Zn.std(0)+1e-6\ns = 0.1\nsave_edit('bert.pooler.dense.weight', np.diag(s/sd).astype(np.float32))\nsave_edit('bert.pooler.dense.bias', (-s*mu/sd).astype(np.float32))\nprint('max |z|', np.abs((Zn-mu)/sd).max(), 'sd range', sd.min(), sd.max())\nprint(len(edits()), 'edits')\n"
}
```

